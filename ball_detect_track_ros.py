#!/usr/bin/env python
from __future__ import print_function

# import roslib
# roslib.load_manifest('my_package')
import sys
# import rospy
import cv2
import numpy as np
# from std_msgs.msg import String
# from MBZIRC_control.msg import ball_xyz
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
# import imutils
import math
import time
from drone_track import droneDetector

class ballTracker:
	# def __init__(self, image_pub, configPath, weighPath, metaPath):
	def __init__(self, configPath, weighPath, metaPath):
		# self.pub = pub
		# self.y_pub = y_pub
		# self.image_pub = image_pub
		self.kf = self.create_kf()
		self.lastx, self.lasty, self.lastr = 0.,0.,0.
		self.lastX, self.lastY, self.lastR = 0, 0, 0
		self.droneminx, self.droneminy, self.dronemaxx, self.dronemaxy = 0, 0, 0, 0
		self.init = 0
		self.detected = 0
		self.prev = 0
		self.found=0
		self.trycont = 0
		self.out = cv2.VideoWriter('drone_trial_1.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))
		self.prs = False
		self.drone_detector = droneDetector(configPath, weighPath, metaPath)

	def getDepth(self, detHeight, detX, detY, height, width):

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

		x,y,r = self.tryhough(mask, self.init, 10)
		# print (x,y,r)
		if x==10000:
			if self.trycont:
				#print ("trying for cont")
				x,y,r = self.trycontours(mask, self.init)
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

	

	def initial(self, frame, init):		
		centerx, centery, rad, val = self.detectball(frame, self.init)
		#print (centerx, centery, rad)
		if self.detected==0:
			self.lastX, self.lastY = centerx, centery
		if (centerx<=self.lastX+50) and (centerx>=self.lastX-50) and (centery<=self.lastY+50) and (centery>=self.lastY-50) and self.prev!=10:
			self.prev+=1
		elif (centerx <=(self.lastX+50)) and (centerx>=(self.lastX-50)) and (centery<=(self.lastY+50)) and (centery>=(self.lastY-50)) and self.prev==10:
			if not(centerx==0.0 and centery==0.0):
				self.detected = 1
				# print ("Ball detected, initiating KF\n")
			if self.prev==10 and self.detected==0:
				self.trycont=1 
		elif (centerx<=self.lastX+50) or (centerx>=self.lastX-50) or (centery>=self.lastY+50) or (centery<=self.lastY-50):
			self.prev = 0
		if not val:
			self.detected = 0
		return centerx, centery, rad, val

	def predictionbound(self, frame, init, masked):		
		centerx, centery, rad, val = self.detectball_bounded(frame, self.init, masked)
		return centerx, centery, rad, val

	def drone_bb(self, detections):
		xmins = []
		xmaxs = []
		bb = []
		for detection in detections:
			confidance, x, y, w, h = detection[1], detections[2]
			if ((self.dronemaxx + 130) < x and (self.droneminx - 130) > x):
				xmin, ymin, xmax, ymax = self.drone_detector.convertBack(float(x), float(y), float(w), float(h))
				xmins.append(xmin)
				xmaxs.append(xmax)
			elif confidance > 0.75:
				xmin, ymin, xmax, ymax = self.drone_detector.convertBack(float(x), float(y), float(w), float(h))
				bb.append((xmin,xmax))
		bb.append((min(xmins), max(xmaxs)))

		return bb
		
	def draw_drone(self, detections, frame):
		best_confidance = 0
		best_detection = False
		for detection in detections:
			confidance, x, y, w, h = detection[1], detections[2]
			if (self.dronemaxx + 140) < x and (self.droneminx - 140) > x and confidance > best_confidance:
				xmin, ymin, xmax, ymax = self.drone_detector.convertBack(float(x), float(y), float(w), float(h))
				droneminx_draw = xmin
				droneminy_draw = ymin
				dronemaxx_draw = xmax
				dronemaxy_draw = ymax
				best_confidance = confidance
				best_detection = True

			elif confidance > max(best_confidance, 0.75):
				droneminx_draw = xmin
				droneminy_draw = ymin
				dronemaxx_draw = xmax
				dronemaxy_draw = ymax
				best_confidance = confidance
				best_detection = True

			else:
				self.droneminx = xmin
				self.droneminy = ymin
				self.dronemaxx = xmax
				self.dronemaxy = ymax
		if detections:
			self.drone_detector.cvDrawBoxes(b'drone', best_confidance, droneminx_draw, droneminy_draw, dronemaxx_draw, dronemaxy_draw, frame) 
			if best_detection:
				self.droneminx = droneminx_draw
				self.droneminy = droneminy_draw
				self.dronemaxx = dronemaxx_draw
				self.dronemaxy = dronemaxy_draw		

		return
		
	def drone_detectball(self, detections, frame):
		drone_bb = drone_bb(detections)
		mask = drone_mask(drone_bb)

		return drone_circles(frame, mask)
	
	def drone_mask(self, drone_bbs):
		masked = np.zeros((480, 640), dtype=np.uint8)
		for drone_bb in drone_bbs:
			masked[:, drone_bb[0]:drone_bb[1]] = 255

		return masked

	def drone_circles(self, frame, masked):
		
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
		a = hsv[:,:,1]

		lower = np.array([140])
		upper = np.array([235])

		#lower2 = np.array([0,50,50])
		#upper2 = np.array([20,255,255])

		mask = cv2.inRange(a, lower, upper)
		kernel = np.ones((3,3),np.uint8)
		mask = cv2.erode(mask, kernel, iterations=1)
		mask = cv2.dilate(mask, kernel, iterations=2)
		mask = cv2.erode(mask, kernel, iterations=1)
		# print (masked.shape)
		maskm = cv2.bitwise_and(mask, mask, mask=masked)
		#cv2.imshow('Mask', a)	
		#cv2.waitKey(1)
		
		x,y,r = self.tryhough(maskm, self.init, 5)
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

		dont=0
		# vid = cv2.VideoCapture(0)
		vid = cv2.VideoCapture('/home/aruul/Pictures/Vid/GOPR1844.MP4')
		total=0 
		dropped=0 
		recent=0
		trust=1
		self.drone_detector.initYOLO()
		predictedCoords = np.zeros((2, 1), np.float32)
		centerx, centery = 0, 0
		cal_x, cal_y = 0, 0 
		# wr_c = 0
		# while not rospy.is_shutdown():
		while True:
			ret, frame = vid.read()
			#frame = frame[:,0:900]
			#time.sleep(0.1)
			if ret==True:
				frame = cv2.resize(frame, (640,480))
				detections = self.drone_detector.drone_detect(frame)
				
				height, width = frame.shape[0], frame.shape[1]
				print ("trust", trust)
				total+=1
				if self.detected:
					print ("detected something")
					masked = self.createMask(predictedCoords, trust, self.lastR)
					centerx, centery, rad, val = self.predictionbound(frame, self.detected, masked)
					cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), rad, [0,255,255], 2, 8)
					cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15), (predictedCoords[0] + 50, predictedCoords[1] - 30), [100, 10, 255], 2, 8)
					cv2.putText(frame, "Predicted", (predictedCoords[0] + 50, predictedCoords[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])
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
					centerx, centery, rad, val = self.initial(frame, self.detected)			
					dont=0
					self.found+=1
					if centerx == 10000 or val==0 or (centerx==0.0 and centery==0.0):
						self.found-=1
						dont=1
						trust+=1
						if trust>20:
							predictedCoords = np.zeros((2, 1), np.float32)
							self.detected=0
							
				if self.detected==0:
					centerx, centery, rad, val = drone_detectball(detections, frame)
					dont=0
					self.found+=1
					if centerx == 10000 or val==0 or (centerx==0.0 and centery==0.0):
						self.found-=1
						dont=1
						trust+=1
						if trust>20:
							predictedCoords = np.zeros((2, 1), np.float32)
							self.detected=0


				if total<15:
					
					if dont==0:
						self.lastX, self.lastY, self.lastR = centerx, centery, rad
						predictedCoords = self.Estimate(centerx, centery)
						trust=1

						centerx, centery, rad = int(centerx), int(centery), int(rad)
						print ("undetected  ", centerx, "  ", centery)
						depth, cal_x, cal_y = self.getDepth(rad, centerx, centery, height, width)
						#use depth, cal_x, cal_y to get (z,x,y)
						cv2.putText(frame, "Depth: %f"  %depth, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])
						cv2.circle(frame, (centerx, centery), rad, [0,0,255], 2, 8)
						cv2.line(frame,(centerx, centery + 20), (centerx + 50, centery + 20), [100,100,255], 2,8)
						cv2.putText(frame, "Actual", (centerx + 50, centery + 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])

				else:
					if dont==0:
						if (centerx>=self.lastX-100) and (centerx<=self.lastX+100) and (centery>=self.lastY-100) and (centery<=self.lastY+100):
							trust=1	
							self.lastX, self.lastY, self.lastR = centerx, centery, rad
							predictedCoords = self.Estimate(centerx, centery)
							print("tracked", centerx, "  ", centery)

							depth, cal_x, cal_y = self.getDepth(rad, centerx, centery, height, width)
							#use depth, cal_x, cal_y to get (z,x,y)
							cv2.putText(frame, "Depth: %f"  %depth, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])
							cv2.circle(frame, (centerx, centery), rad, [0,0,255], 2, 8)
							cv2.line(frame,(centerx, centery + 20), (centerx + 50, centery + 20), [100,100,255], 2,8)
							cv2.putText(frame, "Actual", (centerx + 50, centery + 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])

				frame = self.draw_drone(detections, frame)		
				self.out.write(frame)
				# newimg =ball_xyz()
				# newimg.centerx = centerx
				# newimg.centery = centery
				# newimg.radius = rad		
				cv2.imshow('Input', frame)
				cv2.waitKey(1)
				#use depth, cal_x, cal_y to get (z,x,y)
				# try:
				#   self.image_pub.publish(newimg)
				# except CvBridgeError as e:
				#   print(e)

			else:
				dropped+=1
				if dropped>100:
					break

		print (total)
		print (self.found)
		vid.release()
		cv2.destroyAllWindows()

def main():
	# rospy.init_node('balltrack', anonymous=True)
	# image_pub = rospy.Publisher("image_topic_2",ball_xyz, queue_size=1)
	configPath = '/home/aruul/Videos/Drone-Tracking/yolov3-tiny.cfg'
	weightPath = '/home/aruul/Videos/Drone-Tracking/Net/yolov3-tiny_best.weights'
	metaPath = '/home/aruul/Videos/Drone-Tracking/drone.data'
	ball_Tracker = ballTracker(configPath, weightPath, metaPath)
	ball_Tracker.track()

if __name__=="__main__":
	main()
	#TODO Test new tracker
	#TODO Calculate Accuracy
	#TODO Convert to world coordinates
