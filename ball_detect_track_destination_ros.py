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
y_pub = rospy.Publisher('y_graph', Float32, queue_size = 5)
discard_samples = 20                        #samples to discard before gps normalizes
pos = Point()
quat = Quaternion()
pos.x = pos.y = pos.z = 0
quat.x = quat.y = quat.z = quat.w = 0
start_y = 0.0
timer = 0.0
cached_var = {}

def getDepth(detHeight, detX, detY, height, width):

	#detHeight *= 2
	#f = 0.28
	f = 0.397 #gopro 5
	#f = 0.400
	ballHeight = 2*7.96
	sensorHeight = 0.455 #gopro 5
	#sensorHeight = 0.358 #logitech c270
	sensorWidth = 0.617 #gopro 5

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

class KalmanFilter:

	kf = cv2.KalmanFilter(4, 2)
	kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
	kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)	
	kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03

	def Estimate(self, coordX, coordY):
		measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
		self.kf.correct(measured)
		predicted = self.kf.predict()
		return predicted

lastx, lasty, lastr = 0.,0.,0.
lastX, lastY, lastR = 0, 0, 0
init = 0
detected = 0
prev = 0
found=0
trycont = 0

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

  #   if home_xy_recorded is False and cart_x != 0 and cart_y != 0:
  #       home_x = cart_x
  #       home_y = cart_y
  #       discard_samples = discard_samples - 1

  #   if(discard_samples <= 0):
  #       desired_x = cart_x                          #to set home position as initial desired position
  #       desired_y = cart_y
  #       start_y = home_yaw
		# home_xy_recorded = True

def transform(x, y, z, X, Y, Z, roll_x, pitch_y, yaw_z):

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


def contourShape(cx, cy):

	global lastx, lasty
	print (cx)
	print (cy)
	print (lastx)
	print (lasty)
	print (math.sqrt(math.pow((cx-lastx),2) + math.pow((cy-lasty),2)))
	return math.sqrt(math.pow((cx-lastx),2) + math.pow((cy-lasty),2))

def trycontours(mask, init):

	global lastx, lasty, lastr
	_, contours, __ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
		if init==0:
			lastx, lasty = cx,cy
			return cx,cy,r
			#print ("Contour Success")

		else:
			if (cx<=lastx+50) or (cx>=lastx-50):
				if (cy<=lasty+50) or (cy>=lasty-50):
					lastx, lasty = cx,cy
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
		

def tryhough(mask, init, constraint):

	global lastx, lasty, lastr, found
	circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, constraint, 500, param1=500, param2=70, minRadius=0, maxRadius=500)
	try:
		if circles.any() != None:
			circles = np.uint16(np.around(circles))
			for i in circles[0,:]:
				#cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
				#cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
				if i[0]!=0 and i[1]!=0:
					if init==0:
						lastx, lasty = i[0],i[1]
						found+=1
						return i[0],i[1],i[2]
						#print (i[0],i[1])
						#print ("Hough Success")
					else:
						if (i[0]<=lastx+50) or (i[0]>=lastx-50):
							if (i[1]<=lasty+50) or (i[1]>=lasty-50):
								lastx, lasty = i[0],i[1]
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


def detectball(frame, init):

	global lastx, lasty, lastr, found, trycont
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

	x,y,r = tryhough(mask, init, 10)
	print (x,y,r)
	if x==10000:
		if trycont:
			#print ("trying for cont")
			x,y,r = trycontours(mask, init)
			if x!=10000:
				lastx,lasty,lastr=x,y,r
				return x,y,r,1
			else:
				return lastx,lasty,lastr, 0
		return lastx, lasty, lastr, 0
	else:
		lastx,lasty,lastr=x,y,r
		found+=1
		return x,y,r,1
		

def detectball_bounded(frame, init, masked):

	global lastx, lasty, lastr, found
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
	#print (mask.shape)
	maskm = cv2.bitwise_and(mask, mask, mask=masked)
	#cv2.imshow('Mask', a)	
	#cv2.waitKey(1)
	
	x,y,r = tryhough(maskm, init, 15)
	#print ("here"+str(x)+str(y))
	if x==10000:
		x,y,r = trycontours(mask, init)
		#print ("here22"+str(x)+str(y))
		if x!=10000:
			lastx,lasty,lastr=x,y,r
			found+=1
			return x,y,r,1
		else:
			#print ("here3"+str(lastx)+str(lasty))
			return lastx,lasty,lastr,0
	else:
		lastx,lasty,lastr=x,y,r
		found+=1
		return x,y,r,1



def initial(frame, init):
	
	global lastX, lastY, lastR, detected, prev, trycont
	
	centerx, centery, rad, val = detectball(frame, init)
	#print (centerx, centery, rad)

	if detected==0:
		lastX, lastY = centerx, centery
	if (centerx<=lastX+50) and (centerx>=lastX-50) and (centery<=lastY+50) and (centery>=lastY-50) and prev!=10:
		prev+=1
	elif (centerx <=(lastX+50)) and (centerx>=(lastX-50)) and (centery<=(lastY+50)) and (centery>=(lastY-50)) and prev==10:
		if not(centerx==0.0 and centery==0.0):
			detected = 1
			print ("Ball detected, initiating KF\n")
		if prev==10 and detected==0:
			trycont=1 
	elif (centerx<=lastX+50) or (centerx>=lastX-50) or (centery>=lastY+50) or (centery<=lastY-50):
		prev = 0
	return centerx, centery, rad, val
	print (centerx, centery)

def predictionbound(frame, init, masked):

	global lastX, lastY, lastR
	
	centerx, centery, rad, val = detectball_bounded(frame, init, masked)
	return centerx, centery, rad, val


def createMask(predictedCoords, frame, trust, rad):
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

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('drone_trial_1.avi',fourcc, 20.0, (640,480))

def track():

	global init, lastX, lastY, lastR, detected, found, image_pub
	dont=0
	vid = cv2.VideoCapture(0)
	total=0 
	found=0
	dropped=0 
	recent=0
	trust=1

	kfObj = KalmanFilter()
	predictedCoords = np.zeros((2, 1), np.float32)
	centerx, centery = 0, 0
	cal_x, cal_y = 0 

	global home_xy_recorded, home_z_recorded, cart_x, cart_y, cart_z, desired_x, desired_y, desired_z, home_yaw
    global home_x, home_z, home_y, limit_x, limit_y, limit_z, kp, kb, roll, pitch, yaw, cont, gps_rate, n, t, timer, cached_var
    rospy.init_node('MAVROS_Listener')
    
    rate = rospy.Rate(100.0)

    rospy.Subscriber("/mavros/imu/data", Imu, imu_cb)
    # rospy.Subscriber("/mavros/global_position/global", NavSatFix, gps_global_cb)
	if(gps_rate == 0):
		rospy.Subscriber("/mavros/global_position/local", Odometry, gps_local_cb)

	elif(gps_rate == 1):    
        rospy.Subscriber("/global_position_slow", Odometry, gps_local_cb)

	else:
        gps_rate = 0

    pub = rospy.Publisher('destination_point', PointStamped, queue_size = 1)
    
	while not rospy.is_shutdown():
		ret, frame = vid.read()
		#frame = frame[:,0:900]
		#time.sleep(0.1)
		height, width = frame.shape[0], frame.shape[1]
		cv2.waitKey(1)
		if ret==True:
			print ("trust", trust)
			total+=1
			if detected:
				print ("detected something")
				masked = createMask(predictedCoords, frame, trust, lastR)
				centerx, centery, rad, val = predictionbound(frame, detected, masked)
				cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), rad, [0,255,255], 2, 8)
				cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15), (predictedCoords[0] + 50, predictedCoords[1] - 30), [100, 10, 255], 2, 8)
				cv2.putText(frame, "Predicted", (predictedCoords[0] + 50, predictedCoords[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])
				dont=0
				found+=1
				#print ("gotten centerx", centerx)
				if centerx==10000 or val==0 or (centerx==0.0 and centery==0.0):
					found-=1
					dont=1
					trust+=1
					if trust>20:
						predictedCoords = np.zeros((2, 1), np.float32)
						detected=0

			else:
				print ("still trying")
				centerx, centery, rad, val = initial(frame, detected)			
				dont=0
				found+=1
				if centerx == 10000 or val==0 or (centerx==0.0 and centery==0.0):
					found-=1
					dont=1
					trust+=1
					if trust>20:
						predictedCoords = np.zeros((2, 1), np.float32)
						detected=0

			if total<15 or detected==0:
				
				if dont==0:
					lastX, lastY, lastR = centerx, centery, rad
					predictedCoords = kfObj.Estimate(centerx, centery)
					trust=1

					centerx, centery, rad = int(centerx), int(centery), int(rad)
					print ("undetected  ", centerx, "  ", centery)
					cal_x, cal_y, cal_z = getDepth(rad, centerx, centery, height, width)
					
					transformed_x, transformed_y, transformed_z = transform(cal_x, cal_y, cal_z, cart_x, cart_y, cart_z, roll, pitch, yaw)
					desired_point = PointStamped(header=Header(stamp=rospy.get_rostime()))
					desired_point.header.frame_id = 'target_location'
			        desired_point.point.x = transformed_x
			        desired_point.point.y = transformed_y
			        desired_point.point.z = transformed_z
			        pub.publish(desired_point)


					cv2.putText(frame, "Depth: %f"  %depth, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])
					cv2.circle(frame, (centerx, centery), rad, [0,0,255], 2, 8)
					cv2.line(frame,(centerx, centery + 20), (centerx + 50, centery + 20), [100,100,255], 2,8)
					cv2.putText(frame, "Actual", (centerx + 50, centery + 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])	

			else:
				if dont==0:
					if (centerx>=lastX-100) and (centerx<=lastX+100) and (centery>=lastY-100) and (centery<=lastY+100):
						trust=1	
						lastX, lastY, lastR = centerx, centery, rad
						predictedCoords = kfObj.Estimate(centerx, centery)
						print("tracked", centerx, "  ", centery)

						cal_x, cal_y, cal_z = getDepth(rad, centerx, centery, height, width)

						transformed_x, transformed_y, transformed_z = transform(cal_x, cal_y, cal_z, cart_x, cart_y, cart_z, roll, pitch, yaw)
						desired_point = PointStamped(header=Header(stamp=rospy.get_rostime()))
						desired_point.header.frame_id = 'target_location'
				        desired_point.point.x = transformed_x
				        desired_point.point.y = transformed_y
				        desired_point.point.z = transformed_z
				        pub.publish(desired_point)	
							
						cv2.putText(frame, "Depth: %f"  %depth, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])
						cv2.circle(frame, (centerx, centery), rad, [0,0,255], 2, 8)
						cv2.line(frame,(centerx, centery + 20), (centerx + 50, centery + 20), [100,100,255], 2,8)
						cv2.putText(frame, "Actual", (centerx + 50, centery + 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])

					
			out.write(frame)
			newimg =ball_xyz()
			newimg.centerx = centerx
			newimg.centery = centery
			newimg.radius = rad		
			# cv2.imshow('Input', frame)
			cv2.waitKey(1)
			#use depth, cal_x, cal_y to get (z,x,y)
			try:
			  image_pub.publish(newimg)
			except CvBridgeError as e:
			  print(e)

		else:
			dropped+=1
			if dropped>100:
				break

	print (total)
	print (found)
	vid.release()
	cv2.destroyAllWindows()

def main():
	global image_pub
	rospy.init_node('balltrack', anonymous=True)
	image_pub = rospy.Publisher("image_topic_2",ball_xyz, queue_size=1)
	track()

if __name__=="__main__":
	main()
