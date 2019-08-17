import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import math
import time

def getDepth(detHeight, detX, detY, height, width):

	#detHeight *= 2
	#f = 0.28
	f = 0.397 #gopro 5
	#f = 0.400
	ballHeight = 2*7.96
	sensorHeight = 0.455 #gopro 5
	#sensorHeight = 0.358 #logitech c270
	sensorWidth = 0.617 #gopro 5

	center_pixel_x, center_pixel_y = 450, 360
	detX = (detX - center_pixel_x)*sensorWidth/width
	detY = (detY - center_pixel_y)*sensorHeight/height

	if detHeight!=0:
		depth = f*ballHeight*480/(detHeight*sensorHeight)
		trans_x = detX*depth/f
		trans_y = detY*depth/f
		#trans_y = detY*ballHeight*480/(detHeight*sensorHeight)
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
	hsv = hsv[:,:,1]
	lower = np.array([150])
	upper = np.array([255])

	#lower = np.array([0,50,50])
	#upper = np.array([20,255,255])

	mask = cv2.inRange(hsv, lower, upper)
	kernel = np.ones((3,3),np.uint8)
	mask = cv2.erode(mask, kernel, iterations=1)
	mask = cv2.dilate(mask, kernel, iterations=4)
	mask = cv2.erode(mask, kernel, iterations=1)
	cv2.imshow('Mask', mask)	
	cv2.waitKey(10)

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
	hsv = hsv[:,:,1]
	lower = np.array([150])
	upper = np.array([255])

	#lower2 = np.array([0,50,50])
	#upper2 = np.array([20,255,255])

	mask = cv2.inRange(hsv, lower, upper)
	kernel = np.ones((3,3),np.uint8)
	mask = cv2.erode(mask, kernel, iterations=1)
	mask = cv2.dilate(mask, kernel, iterations=6)
	mask = cv2.erode(mask, kernel, iterations=3)
	#print (mask.shape)
	maskm = cv2.bitwise_and(mask, mask, mask=masked)
	cv2.imshow('Mask', mask)	
	cv2.waitKey(10)
	
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
	masked = np.zeros((480,856), dtype=np.uint8)
	for i in range(int(predictedCoords[0])-a, int(predictedCoords[0])+a):
		for j in range(int(predictedCoords[1])-a, int(predictedCoords[1])+a):
			try:
				masked[j][i]=255
			except IndexError:
				pass
	cv2.imshow('mmm',masked)
	return masked

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('pre_2.avi',fourcc, 20.0, (900,720))

def track():

	global init, lastX, lastY, lastR, detected, found
	dont=0
	vid = cv2.VideoCapture('/home/shantam/Documents/Programs/PoseEstimation/gopro.mp4')
	total=0 
	found=0
	dropped=0 
	recent=0
	trust=1

	kfObj = KalmanFilter()
	predictedCoords = np.zeros((2, 1), np.float32)
	centerx, centery = 0, 0 

	count=1000	
	while vid.isOpened():
		#ret, frame = vid.read()
		count+=1
		ret=True
		frame = cv2.imread('/home/shantam/ros_ws/Drone_GPS/1/frame'+str(count)+'.jpg')
		#frame = frame[:,0:900]
		height, width = frame.shape[0], frame.shape[1]	
		print (height, width)
		#time.sleep(0.1)
		cv2.waitKey(5)
		if ret==True:
			print ("trust", trust)
			total+=1
			if detected:
				masked = createMask(predictedCoords, frame, trust, lastR)
				centerx, centery, rad, val = predictionbound(frame, detected, masked)
				cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), rad, [0,255,255], 2, 8)
				cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15), (predictedCoords[0] + 50, predictedCoords[1] - 30), [100, 10, 255], 2, 8)
				cv2.putText(frame, "Predicted", (predictedCoords[0] + 50, predictedCoords[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])
				dont=0
				found+=1
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
					depth, trans_x, trans_y = getDepth(rad, centerx, centery, height, width)
					cv2.putText(frame, "Depth: %f"  %depth, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,0,0])
					cv2.putText(frame, "X: %f"  %trans_x, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,0,0])
					cv2.putText(frame, "Y: %f"  %trans_y, (50, 150), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,0,0])
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

						depth, trans_x, trans_y = getDepth(rad, centerx, centery, height, width)
						cv2.putText(frame, "Depth: %f"  %depth, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,0,0])
						cv2.putText(frame, "X: %f"  %trans_x, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,0,0])
						cv2.putText(frame, "Y: %f"  %trans_y, (50, 150), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,0,0])
						cv2.circle(frame, (centerx, centery), rad, [0,0,255], 2, 8)
						cv2.line(frame,(centerx, centery + 20), (centerx + 50, centery + 20), [100,100,255], 2,8)
						cv2.putText(frame, "Actual", (centerx + 50, centery + 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])
					
					
			out.write(frame)
			cv2.imshow('Input', frame)
			cv2.waitKey(1)

		else:
			dropped+=1
			if dropped>100:
				break

	print (total)
	print (found)
	vid.release()
	cv2.destroyAllWindows()

def main():
	track()

if __name__=="__main__":
	main()
