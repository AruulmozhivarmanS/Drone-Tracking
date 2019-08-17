#!/usr/bin/env python
from __future__ import print_function

import roslib
# roslib.load_manifest('my_package')
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from MBZIRC_control.msg import ball_xyz
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import argparse
# import imutils
import math
import time
from drone_track import droneDetector
import os, sys, select, termios, tty, roslib, rospy, argparse, mavros, threading, time, readline, signal, tf

from sensor_msgs.msg import Joy
from std_msgs.msg import Header, Float32, Float64, Empty
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3, Quaternion, Point, Twist, PointStamped
from subprocess import call
from mavros_msgs.msg import OverrideRCIn
from mavros import command
from mavros import setpoint as SP
from mavros_msgs.msg import PositionTarget
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import Marker

from qp_planner.msg import algomsg
from mavros_msgs.msg import Altitude
from gazebo_msgs.msg import ModelStates

from sensor_msgs.msg import Imu, NavSatFix
from tf.transformations import euler_from_quaternion
import quadprog
from numpy import array
from qp_matrix import qp_q_dot_des_array
from MPC import MPC_solver
import qp_matrix

def imu_cb(data):
	global roll, pitch, yaw
	orientation_q = data.orientation
	orientation_list = (orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
	(roll, pitch, yaw) = euler_from_quaternion(orientation_list)
#(roll, pitch, yaw) = (roll * 180.0/3.1416, pitch * 180.0/3.1416, yaw  * 180.0/3.1416)

def gps_local_cb(data):
	global cart_x, cart_y, home_x, home_y, home_xy_recorded, discard_samples, desired_x, desired_y, start_y

	cart_x = data.pose.pose.position.x
	cart_y = data.pose.pose.position.y
	cart_z = data.pose.pose.position.z

global R
global roll, pitch, yaw

n = 15
t = 0.1
gps_rate = 0
cont = 0
home_xy_recorded = home_z_recorded = False
cart_x = cart_y = cart_z = 0.0
home_x = home_y = home_z = 0.0
ekf_x = ekf_y = ekf_z = 0
desired_x = desired_y =  desired_z = 0.0
limit_x = limit_y = limit_z = 10.
roll = pitch = yaw = 0.0
TIMEOUT = 0.5
kp = 1.
kb = 10000000.0
home_yaw = 0
br = tf.TransformBroadcaster()
br2 = tf.TransformBroadcaster()
discard_samples = 20                        #samples to discard before gps normalizes
pos = Point()
quat = Quaternion()
pos.x = pos.y = pos.z = 0
quat.x = quat.y = quat.z = quat.w = 0
start_y = 0.0
timer = 0.0
cached_var = {}

class ballTracker:
	def __init__(self, image_pub, y_pub, pub, configPath, weighPath, metaPath):
	# def __init__(self, configPath, weighPath, metaPath):
		self.pub = pub
		self.y_pub = y_pub
		self.image_pub = image_pub
		self.kf = self.create_kf()
		self.lastx, self.lasty, self.lastr = 0.,0.,0.
		self.lastX, self.lastY, self.lastR = 0, 0, 0
		self.droneminx, self.droneminy, self.dronemaxx, self.dronemaxy = 0, 0, 0, 0
		self.init = 0
		self.detected = 0
		self.prev = 0
		self.found=0
		self.trycont = 0
		self.out = cv2.VideoWriter('drone_trial_2.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))
		self.prs = False
		self.drone_detector = droneDetector(configPath, weighPath, metaPath)
		# self.im_save_count = 0
		self.pos_x = 0
		self.pos_y = 0
		self.pos_z = 0

	def getDepth(self, detHeight, detX, detY, height, width):

		#detHeight *= 2
		#f = 0.28
		# f = 0.397 #gopro 5
		f = 0.5
		#f = 0.400
		ballHeight = 2*7.96
		# sensorHeight = 0.455 #gopro 5
		sensorHeight = 0.5
		#sensorHeight = 0.358 #logitech c270
		# sensorWidth = 0.617 #gopro 5
		sensorWidth	= 0.5

		center_pixel_x, center_pixel_y = width/2, height/2
		detX = (detX - center_pixel_x)*sensorWidth/width
		detY = (detY - center_pixel_y)*sensorHeight/height

		if detHeight!=0:
			depth = f*ballHeight*480/(detHeight*sensorHeight)
			trans_x = detX*depth/f
			trans_y = detY*depth/f
		else:
			depth=0
			trans_x=0
			trans_y=0

		return depth, trans_x, trans_y

	def create_kf(self):
		kf = cv2.KalmanFilter(4, 2)
		kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
		kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)	
		kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
		return kf

	def Estimate(self, coordX, coordY):
		measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
		self.kf.correct(measured)
		predicted = self.kf.predict()
		
		return predicted

	def contourShape(self, cx, cy):
		
		return math.sqrt(math.pow((cx-self.lastx),2) + math.pow((cy-self.lasty),2))

	def trycontours(self, mask, init):
		contours, __ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		minDist = 1000000
		maxHeight = 100000
		order = 0	

		if len(contours)>1:
			for i in range(len(contours)):
				#print ('Contour Alert')
				#time.sleep(0.1)
				cnt = contours[i]
				M = cv2.moments(cnt)
				if M['m00']==0:
					continue
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				#dist = contourShape(cx, cy)
				if cy<maxHeight:
					maxHeight = cy
					order = i

		try:
			cnt = contours[order]
			M = cv2.moments(cnt)
			if M['m00']==0:
				return 10000,10000,10000	
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			#(x,y),r = cv2.minEnclosingCircle(cnt)
			#center = (int(x),int(y))
			area=cv2.contourArea(cnt)
			r = int(math.sqrt(area/np.pi))
		except IndexError:
			print("contour index failure", order)
			return 10000,10000,100000

		if cx!=0 and cy!=0:
			if self.init==0:
				self.lastx, self.lasty = cx,cy
				return cx,cy,r
				#print ("Contour Success")

			else:
				if (cx<=self.lastx+50) or (cx>=self.lastx-50):
					if (cy<=self.lasty+50) or (cy>=self.lasty-50):
						self.lastx, self.lasty = cx,cy
						return cx,cy,r
						#print ("contour Success")
					else:
						print ("contour outside reliable area")
						return 10000,10000,10000
				else:
					return 10000,10000,10000
		else:
			#print ("No circles detected via Contours")
			return 10000,10000,10000

	def tryhough(self, mask, init, constraint):
		circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, constraint, 500, param1=500, param2=70, minRadius=0, maxRadius=500)
		# print(circles)
		try:
			if circles.any() != None:
				circles = np.uint16(np.around(circles))
				for i in circles[0,:]:
					#cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
					#cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
					if i[0]!=0 and i[1]!=0:
						if self.init==0:
							self.lastx, self.lasty = i[0],i[1]
							self.found+=1
							return i[0],i[1],i[2]
							#print (i[0],i[1])
							#print ("Hough Success")
						else:
							if (i[0]<=self.lastx+50) or (i[0]>=self.lastx-50):
								if (i[1]<=self.lasty+50) or (i[1]>=self.lasty-50):
									self.lastx, self.lasty = i[0],i[1]
									#found+=1
									#print ("Hough Success")
									#print (i[0],i[1])
									return i[0],i[1],i[2]
							else:
								return 10000,10000,10000
					else:
						return 10000,10000,10000
		except AttributeError:
			#print ("No circles found via Hough")
			#return lastx, lasty
			return 10000,10000,10000

	def tryhough_drone(self, mask, init, constraint):
		circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, constraint, 400, param1=480, param2=21, minRadius=0, maxRadius=500)
		print('Circles')
		print(circles)
		try:
			if circles.any() != None:
				circles = np.uint16(np.around(circles))
				for i in circles[0,:]:
					#cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
					#cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
					if i[0]!=0 and i[1]!=0:
						if self.init==0:
							self.lastx, self.lasty = i[0],i[1]
							self.found+=1
							return i[0],i[1],i[2]
							#print (i[0],i[1])
							#print ("Hough Success")
						else:
							if (i[0]<=self.lastx+50) or (i[0]>=self.lastx-50):
								if (i[1]<=self.lasty+50) or (i[1]>=self.lasty-50):
									self.lastx, self.lasty = i[0],i[1]
									#found+=1
									#print ("Hough Success")
									#print (i[0],i[1])
									return i[0],i[1],i[2]
							else:
								return 10000,10000,10000
					else:
						return 10000,10000,10000
		except AttributeError:
			#print ("No circles found via Hough")
			#return lastx, lasty
			return 10000,10000,10000

	def detectball(self, frame, init):
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
		a = hsv[:,:,1]

		lower = np.array([160])
		upper = np.array([200])

		# lower = np.array([0,100,100])
		# upper = np.array([10,255,255])

		#lower = np.array([10,50,50])
		#upper = np.array([20,255,255])

		mask = cv2.inRange(a, lower, upper)
		kernel = np.ones((3,3),np.uint8)
		mask = cv2.erode(mask, kernel, iterations=1)
		mask = cv2.dilate(mask, kernel, iterations=4)
		mask = cv2.erode(mask, kernel, iterations=1)
		#cv2.imshow('Masked', a)	
		#cv2.waitKey(1)

		x,y,r = self.tryhough(mask, self.init, 12)
		# print (x,y,r)
		if x==10000:
			if self.trycont:
				#print ("trying for cont")
				x,y,r = self.trycontours(mask, self.init)
				# print (x,y,r)
				if x!=10000:
					self.lastx, self.lasty, self.lastr=x,y,r
					return x,y,r,1
				else:
					return self.lastx, self.lasty, self.lastr, 0
			return self.lastx, self.lasty, self.lastr, 0
		else:
			self.lastx, self.lasty, self.lastr=x,y,r
			self.found+=1
			return x,y,r,1

	def detectball_bounded(self, frame, init, masked):

		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
		a = hsv[:,:,1]

		lower = np.array([160])
		upper = np.array([200])

		#lower2 = np.array([0,50,50])
		#upper2 = np.array([20,255,255])

		mask = cv2.inRange(a, lower, upper)
		kernel = np.ones((3,3),np.uint8)
		mask = cv2.erode(mask, kernel, iterations=1)
		mask = cv2.dilate(mask, kernel, iterations=6)
		mask = cv2.erode(mask, kernel, iterations=3)
		# print (masked.shape)
		maskm = cv2.bitwise_and(mask, mask, mask=masked)
		#cv2.imshow('Mask', a)	
		#cv2.waitKey(1)
		
		x,y,r = self.tryhough(maskm, self.init, 15)
		#print ("here"+str(x)+str(y))
		if x==10000:
			x,y,r = self.trycontours(mask, self.init)
			#print ("here22"+str(x)+str(y))
			if x!=10000:
				self.lastx, self.lasty, self.lastr=x,y,r
				self.found+=1
				return x,y,r,1
			else:
				#print ("here3"+str(lastx)+str(lasty))
				return self.lastx, self.lasty, self.lastr,0
		else:
			self.lastx, self.lasty, self.lastr=x,y,r
			self.found+=1
			return x,y,r,1

	def transform(self, x, y, z, X, Y, Z, roll_x, pitch_y, yaw_z):

		#x,y,z are destination coordinates
		#X,Y,Z are drone coordinates with respect to initial position

		transform_matrix = np.zeros(3,4)
		dest = np.zeros(4,1)
		trans = np.zeros(4,1)

		dest[0], dest[1], dest[2], dest[3] = x, y, z, 1

		transform_matrix[0][0] = cos(yaw_z)*cos(pitch_y)
		transform_matrix[0][1] = cos(yaw_z)*sin(pitch_y)*sin(roll_x) - sin(yaw_z)*cos(roll_x)
		transform_matrix[0][2] = sin(yaw_z)*sin(pitch_y)*cos(roll_x) - sin(roll_x)*cos(yaw_z)
		transform_matrix[0][3] = X

		transform_matrix[1][0] = sin(yaw_z)*cos(pitch_y)
		transform_matrix[1][1] = sin(yaw_z)*sin(pitch_y)*sin(roll_x) + cos(yaw_z)*cos(roll_x)
		transform_matrix[1][2] = sin(yaw_z)*sin(pitch_y)*cos(roll_x) - cos(yaw_z)*sin(roll_x)
		transform_matrix[1][3] = Y

		transform_matrix[2][0] = -sin(pitch_y)
		transform_matrix[2][1] = cos(pitch_y)*sin(roll_x)
		transform_matrix[2][2] = cos(pitch_y)*cos(roll_x)
		transform_matrix[2][3] = Z

		transform_matrix[3][3] = 1

		trans = np.matmul(transform_matrix, dest)

		return(trans[0], trans[1], trans[2])

	def initial(self, frame, init, detections, depth):		
		centerx, centery, rad, val = self.detectball(frame, self.init)
		# print('centerx :' + str(centerx))
		if not val:
			centerx, centery, rad, val = self.drone_detectball(detections, frame, depth)

		#print (centerx, centery, rad)
		if self.detected==0:
			print('Here - 1')
			self.lastX, self.lastY = centerx, centery
		if (centerx<=self.lastX+50) and (centerx>=self.lastX-50) and (centery<=self.lastY+50) and (centery>=self.lastY-50) and self.prev!=10:
			print('Here - 2')
			self.prev+=1
		elif (centerx <=(self.lastX+50)) and (centerx>=(self.lastX-50)) and (centery<=(self.lastY+50)) and (centery>=(self.lastY-50)) and self.prev==10:
			if not(centerx==0.0 and centery==0.0):
				print('Here - 3')
				self.detected = 1
				# print ("Ball detected, initiating KF\n")
			if self.prev==10 and self.detected==0:
				print('Here - 4')
				self.trycont=1 
		elif (centerx<=self.lastX+50) or (centerx>=self.lastX-50) or (centery>=self.lastY+50) or (centery<=self.lastY-50):
			print('Here - 5')
			self.prev = 0

		return centerx, centery, rad, val

	def predictionbound(self, frame, init, masked):		
		centerx, centery, rad, val = self.detectball_bounded(frame, self.init, masked)
		return centerx, centery, rad, val

	def drone_bb(self, detections):
		xmins = []
		xmaxs = []
		ymins = []
		ymaxs = []
		bbx = []
		bby = []
		best_detection = False
		# print(detections)
		# print(self.droneminx, self.dronemaxx)
		for detection in detections:
			confidance, x, y, w, h = detection[1], detection[2][0], detection[2][1], detection[2][2], detection[2][3]
			if (self.dronemaxx and self.droneminx and self.droneminy and self.dronemaxy) == 0:
				xmin, ymin, xmax, ymax = self.drone_detector.convertBack(float(x), float(y), float(w), float(h))
				self.droneminx, self.dronemaxx, self.droneminy, self.dronemaxy = xmin, xmax, ymin, ymax
				# print(self.droneminx, self.dronemaxx)
			if ((self.dronemaxx + 90) > x and (self.droneminx - 90) < x):
				xmin, ymin, xmax, ymax = self.drone_detector.convertBack(float(x), float(y), float(w), float(h))
				xmins.append(xmin)
				xmaxs.append(xmax)
				ymins.append(ymin)
				ymaxs.append(ymax)
				best_detection = True
			elif confidance > 0.95:
				print(confidance)
				print('Im Here')
				xmin, ymin, xmax, ymax = self.drone_detector.convertBack(float(x), float(y), float(w), float(h))
				bbx.append((xmin,xmax))
				bby.append((ymin, ymax))
		if detections and best_detection:
			bbx.append((min(xmins), max(xmaxs)))
			bby.append((min(ymins), max(ymaxs)))
		# print(best_detection)
		# print(bbx, bby)
		return bbx, bby
		
	def draw_drone(self, detections, frame):
		best_confidance = 0
		best_detection_out = 0
		best_detection = False
		for detection in detections:
			confidance, x, y, w, h = detection[1], detection[2][0], detection[2][1], detection[2][2], detection[2][3]
			xmin, ymin, xmax, ymax = self.drone_detector.convertBack(float(x), float(y), float(w), float(h))
			if (self.dronemaxx + 90) > x and (self.droneminx - 90) < x and confidance > best_confidance:
				droneminx_draw = xmin
				droneminy_draw = ymin
				dronemaxx_draw = xmax
				dronemaxy_draw = ymax
				best_confidance = confidance
				best_detection = True

			elif confidance > max(best_confidance, 0.95):
				droneminx_draw = xmin
				droneminy_draw = ymin
				dronemaxx_draw = xmax
				dronemaxy_draw = ymax
				best_confidance = confidance
				best_detection = True

			elif confidance > best_detection_out:
				self.droneminx = xmin
				self.droneminy = ymin
				self.dronemaxx = xmax
				self.dronemaxy = ymax
				best_detection_out = confidance
		if best_detection:
			# cv2.imwrite('./detected/' + str(self.im_save_count) + '.jpg', frame)
			# self.im_save_count += 1
			frame = self.drone_detector.cvDrawBoxes(b'drone', best_confidance, droneminx_draw, droneminy_draw, dronemaxx_draw, dronemaxy_draw, frame) 
			self.droneminx = droneminx_draw
			self.droneminy = droneminy_draw
			self.dronemaxx = dronemaxx_draw
			self.dronemaxy = dronemaxy_draw		
		
		return frame
		
	def drone_detectball(self, detections, frame, depth):
		# print('Heee ' + str(self.im_save_count))
		# self.im_save_count += 1
		bbxs, bbys = self.drone_bb(detections)
		mask = self.drone_mask(bbxs, bbys, depth)
		cv2.imshow('Mask', mask)
		# cv2.waitKey(1)
		x, y, r, val = self.drone_circles(frame, mask)

		return x, y, r, val
	
	def drone_mask(self, bbxs, bbys, depth):
		f = 5
		masked = np.zeros((480, 640), dtype=np.uint8)
		for bbx in bbxs:
			for bby in bbys:
				masked[int(bby[0] * 0.8):int(bby[1] * 1.2 + 2500 * 5/depth), int(bbx[0] * 0.8): int(bbx[1] * 1.2)] = 255
		return masked

	def drone_circles(self, frame, masked):
		
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
		a = hsv[:,:,1]
		cv2.imshow('HSV', a)	
		# cv2.waitKey(1)
		lower = np.array([150])
		upper = np.array([220])

		#lower2 = np.array([0,50,50])
		#upper2 = np.array([20,255,255])

		mask = cv2.inRange(a, lower, upper)
		cv2.imshow('Mask0', mask)
		# cv2.waitKey(1)
		kernel = np.ones((3,3),np.uint8)
		mask = cv2.dilate(mask, kernel, iterations=3)
		mask = cv2.erode(mask, kernel, iterations=1)
		mask = cv2.dilate(mask, kernel, iterations=2)
		mask = cv2.erode(mask, kernel, iterations=1)
		cv2.imshow('Mask1', mask)
		# cv2.waitKey(1)
		print (masked.shape)
		maskm = cv2.bitwise_and(mask, mask, mask=masked)
		cv2.imshow('Maskm', maskm)
		# cv2.waitKey()
		x,y,r = self.tryhough_drone(maskm, self.init, 2)
		# print ("here"+str(x)+str(y))
		# if x != 10000:
		# 	# if (x <=(self.lastX+50)) and (x>=(self.lastX-50)) and (y<=(self.lastY+50)) and (y>=(self.lastY-50)) and self.prev==10 and not(x==0.0 and y==0.0):
		# 	self.lastx, self.lasty, self.lastr = x,y,r
		# 	self.found+=1
		# 	return x,y,r,1
		# else:
		# 	# x,y,r = self.trycontours(mask, self.init)
		# 	# print ("here22"+str(x)+str(y))
		# 	# if x!=10000:
		# 	# 	self.lastx, self.lasty, self.lastr=x,y,r
		# 	# 	self.found+=1
		# 	# 	return x,y,r,1
		# 	# else:
		# 	# 	#print ("here3"+str(lastx)+str(lasty))
		# 	return 10000, 10000, 10000, 0
		if x==10000:
			if self.trycont:
				#print ("trying for cont")
				x,y,r = self.trycontours(mask, self.init)
				# print (x,y,r)
				if x!=10000:
					self.lastx, self.lasty, self.lastr=x,y,r
					print('circle Here')
					print(x, y, r)
					return x,y,r,1
				else:
					return self.lastx, self.lasty, self.lastr, 0
			return self.lastx, self.lasty, self.lastr, 0
		else:
			print('Help')
			print('circle Here')
			print(x, y, r)
			self.lastx, self.lasty, self.lastr=x,y,r
			self.found+=1
			return x,y,r,1

	def createMask(self, predictedCoords, trust, rad):
		a = rad + 30 + int(trust*50*0.1)
		masked = np.zeros((480,640), dtype=np.uint8)
		for i in range(int(predictedCoords[0])-a, int(predictedCoords[0])+a):
			for j in range(int(predictedCoords[1])-a, int(predictedCoords[1])+a):
				try:
					masked[j][i]=255
				except IndexError:
					pass
		#cv2.imshow('mmm',masked)
		return masked
		
	def track(self):
		# a = 0
		# a_c = 0
		# b = 0
		# b_c = 0
		# c = 0
		# c_c = 0
		# d = 0
		# d_c = 0
		global home_xy_recorded, home_z_recorded, cart_x, cart_y, cart_z, desired_x, desired_y, desired_z, home_yaw
   		global home_x, home_z, home_y, limit_x, limit_y, limit_z, kp, kb, roll, pitch, yaw, cont, gps_rate, n, t, timer, cached_var
		dont=0
		depth = 1000
		vid = cv2.VideoCapture(0) # 640x480 px
		# vid = cv2.VideoCapture('./new.mp4') 
		total=0 
		dropped=0 
		recent=0
		trust=1
		FLAG_1 = False
		FLAG_2 = False
		FLAG_3 = False
		self.drone_detector.initYOLO()
		predictedCoords = np.zeros((2, 1), np.float32)
		centerx, centery = 0, 0
		cal_x, cal_y = 0, 0 
		# wr_c = 0
		rate = rospy.Rate(100.0)
		rospy.Subscriber("/mavros/imu/data", Imu, imu_cb)
		# if(gps_rate == 0):
		# 	rospy.Subscriber("/mavros/global_position/local", Odometry, gps_local_cb)
		# elif(gps_rate == 1):    
        # 	rospy.Subscriber("/global_position_slow", Odometry, gps_local_cb)
		# else:
        # 	gps_rate = 0
    	# pub = rospy.Publisher('destination_point', PointStamped, queue_size = 1)
		
		while not rospy.is_shutdown():
		# while True:
			ret, frame = vid.read()
			#frame = frame[:,0:900]
			#time.sleep(0.1)
			if ret==True:
				# frame = cv2.resize(frame, (640,480))
				detections = self.drone_detector.drone_detect(frame)
				
				height, width = frame.shape[0], frame.shape[1]
				print ("trust", trust)
				total+=1
				if self.detected:
					print ("detected something")
					masked = self.createMask(predictedCoords, trust, self.lastR)
					centerx, centery, rad, val = self.predictionbound(frame, self.detected, masked)
					FLAG_1 = True
					dont=0
					self.found+=1
					#print ("gotten centerx", centerx)
					if centerx==10000 or val==0 or (centerx==0.0 and centery==0.0):
						self.found-=1
						dont=1
						trust+=1
						if trust>20:
							predictedCoords = np.zeros((2, 1), np.float32)
							self.detected=0

				else:
					print ("still trying")
					centerx, centery, rad, val = self.initial(frame, self.detected, detections, depth/10)			
					dont=0
					self.found+=1
					if centerx == 10000 or val==0 or (centerx==0.0 and centery==0.0):
						self.found-=1
						dont=1
						trust+=1
						if trust>20:
							predictedCoords = np.zeros((2, 1), np.float32)
							self.detected=0
							
				# if self.detected==0 and total > 15:
					# centerx, centery, rad, val = self.drone_detectball(detections, frame)
					# print('~~~~~~~~~~~~~' + str(centerx) + ' ' + str(centery) + ' ' + str(rad) + '~~~~~~~~~~~~~~~~~~~~~')
					# if centerx == 10000 or val==0 or (centerx==0.0 and centery==0.0):
					# 	self.found -=1
					# 	dont = 1
					# 	trust+=1
					# 	if trust>20:
					# 		predictedCoords = np.zeros((2, 1), np.float32)
					# 		self.detected=0

				if total<15:
					
					if dont==0:
						self.lastX, self.lastY, self.lastR = centerx, centery, rad
						predictedCoords = self.Estimate(centerx, centery)
						trust=1

						centerx, centery, rad = int(centerx), int(centery), int(rad)
						print ("undetected  ", centerx, "  ", centery)
						depth, cal_x, cal_y = self.getDepth(rad, centerx, centery, height, width)
						#use depth, cal_x, cal_y to get (z,x,y)
						FLAG_2 = True

				else:
					if dont==0:
						if (centerx>=self.lastX-100) and (centerx<=self.lastX+100) and (centery>=self.lastY-100) and (centery<=self.lastY+100):
							trust=1	
							self.lastX, self.lastY, self.lastR = centerx, centery, rad
							predictedCoords = self.Estimate(centerx, centery)
							print("tracked", centerx, "  ", centery)

							depth, cal_x, cal_y = self.getDepth(rad, centerx, centery, height, width)
							#use depth, cal_x, cal_y to get (z,x,y)
							FLAG_3 = True
				# if detections:
				if FLAG_1:
					cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), rad, [0,255,255], 2, 8)
					cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15), (predictedCoords[0] + 50, predictedCoords[1] - 30), [100, 10, 255], 2, 8)
					cv2.putText(frame, "Predicted", (predictedCoords[0] + 50, predictedCoords[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])
					
				if FLAG_2:
					cv2.putText(frame, "Depth: %f"  %depth, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])
					cv2.circle(frame, (centerx, centery), rad, [0,0,255], 2, 8)
					cv2.line(frame,(centerx, centery + 20), (centerx + 50, centery + 20), [100,100,255], 2,8)
					cv2.putText(frame, "Actual", (centerx + 50, centery + 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])
					transformed_x, transformed_y, transformed_z = self.transform(cal_x, cal_y, depth, cart_x, cart_y, cart_z, roll, pitch, yaw)
					desired_point = PointStamped(header=Header(stamp=rospy.get_rostime()))
					desired_point.header.frame_id = 'target_location'
			        desired_point.point.x = transformed_x
			        desired_point.point.y = transformed_y
			        desired_point.point.z = transformed_z
			        self.pub.publish(desired_point)

				if FLAG_3:
					transformed_x, transformed_y, transformed_z = self.transform(cal_x, cal_y, depth, cart_x, cart_y, cart_z, roll, pitch, yaw)
					desired_point = PointStamped(header=Header(stamp=rospy.get_rostime()))
					desired_point.header.frame_id = 'target_location'
					desired_point.point.x = transformed_x
					desired_point.point.y = transformed_y
					desired_point.point.z = transformed_z
					self.pub.publish(desired_point)	
					# TODO To be published
					velocity = ((transformed_x - self.pos_x)/(prev_time - time.time()), (transformed_y - self.pos_y)/(prev_time - time.time()), (transformed_z - self.pos_z)/(prev_time - time.time()))	
					cv2.putText(frame, "Depth: %f"  %depth, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])
					cv2.circle(frame, (centerx, centery), rad, [0,0,255], 2, 8)
					cv2.line(frame,(centerx, centery + 20), (centerx + 50, centery + 20), [100,100,255], 2,8)
					cv2.putText(frame, "Actual", (centerx + 50, centery + 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])
				
				frame = self.draw_drone(detections, frame)	
				self.pos_x, self.pos_y, self.pos_z = transformed_x, transformed_y, transformed_z
				prev_time = time.time()
				self.out.write(frame)
				newimg =ball_xyz()
				newimg.centerx = centerx
				newimg.centery = centery
				newimg.radius = rad	
				# if total > 15:
				# 	# if depth < 1000:
				# 	if detections:
				# 		a += 1
				# 	a_c +=1
					# elif depth < 1500:
					# 	if detections:
					# 		b +=1
					# 	b_c +=1
					# elif depth < 2000:
					# 	if detections:
					# 		c += 1
					# 	c_c +=1
					# else:
					# 	if detections:
					# 		d +=1
					# 	d_c +=1

				# newimg =ball_xyz()
				# newimg.centerx = centerx
				# newimg.centery = centery
				# newimg.radius = rad	
				# cv2.imshow('Input', frame)
				cv2.waitKey(1)
				#use depth, cal_x, cal_y to get (z,x,y)
				try:
				  self.image_pub.publish(newimg)
				except CvBridgeError as e:
				  print(e)

			else:
				dropped+=1
				if dropped>100:
					break
		vid.release()
		cv2.destroyAllWindows()
		# print(' : ' + str(a/a_c))
		# print(' : ' + str(a/a_c))
		# print(' : ' + str(a/a_c))
		# print(' : ' + str(a/a_c))
		print (total)
		print (self.found)
		

def main():
	# rospy.init_node('balltrack', anonymous=True)
	# rospy.init_node('MAVROS_Listener')
	# image_pub = rospy.Publisher("image_topic_2",ball_xyz, queue_size=1)
	# y_pub = rospy.Publisher('y_graph', Float32, queue_size = 5)
	configPath = './yolov3-tiny.cfg'
	weightPath = './Net/yolov3-tiny_best.weights'
	metaPath = './drone.data'
	ball_Tracker = ballTracker(image_pub, y_pub, pub, configPath, weightPath, metaPath)
	ball_Tracker.track()

if __name__=="__main__":
	main()