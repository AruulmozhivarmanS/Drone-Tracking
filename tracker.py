from ball_detect_track_ros import ballTracker
from drone_track import droneDetector
import math
import time
import cv2
import numpy as np
import sys

# import roslib
# roslib.load_manifest('my_package')
# import rospy
# from std_msgs.msg import String
# from MBZIRC_control.msg import ball_xyz
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
# import imutils

def main():
    dont=0
    # vid = cv2.VideoCapture(0)
    vid = cv2.VideoCapture('/home/aruul/Pictures/Vid/GOPR1844.MP4')
    total=0 
    found=0
    dropped=0 
    recent=0
    trust=1

    predictedCoords = np.zeros((2, 1), np.float32)
    centerx, centery = 0, 0
    cal_x, cal_y = 0, 0 

    # while not rospy.is_shutdown():
    while True:
        ret, frame = vid.read()
        if ret==True:
            frame = cv2.resize(frame, (640,480))
            height, width = frame.shape[0], frame.shape[1]
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
                    predictedCoords = Estimate(centerx, centery)
                    trust=1

                    centerx, centery, rad = int(centerx), int(centery), int(rad)
                    print ("undetected  ", centerx, "  ", centery)
                    depth, cal_x, cal_y = getDepth(rad, centerx, centery, height, width)
                    #use depth, cal_x, cal_y to get (z,x,y)
                    cv2.putText(frame, "Depth: %f"  %depth, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])
                    cv2.circle(frame, (centerx, centery), rad, [0,0,255], 2, 8)
                    cv2.line(frame,(centerx, centery + 20), (centerx + 50, centery + 20), [100,100,255], 2,8)
                    cv2.putText(frame, "Actual", (centerx + 50, centery + 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])	
            else:
                if dont==0:
                    if (centerx>=lastX-100) and (centerx<=lastX+100) and (centery>=lastY-100) and (centery<=lastY+100):
                        trust=1	
                        lastX, lastY, lastR = centerx, centery, rad
                        predictedCoords = Estimate(centerx, centery)
                        print("tracked", centerx, "  ", centery)

                        depth, cal_x, cal_y = getDepth(rad, centerx, centery, height, width)
                        #use depth, cal_x, cal_y to get (z,x,y)
                        cv2.putText(frame, "Depth: %f"  %depth, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])
                        cv2.circle(frame, (centerx, centery), rad, [0,0,255], 2, 8)
                        cv2.line(frame,(centerx, centery + 20), (centerx + 50, centery + 20), [100,100,255], 2,8)
                        cv2.putText(frame, "Actual", (centerx + 50, centery + 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])

                    
            out.write(frame)
            # newimg =ball_xyz()
            # newimg.centerx = centerx
            # newimg.centery = centery
            # newimg.radius = rad		
            # cv2.imshow('Input', frame)
            # cv2.waitKey(1)
            #use depth, cal_x, cal_y to get (z,x,y)
            # try:
            #   image_pub.publish(newimg)
            # except CvBridgeError as e:
            #   print(e)

        else:
            dropped+=1
            if dropped>100:
                break

    print (total)
    print (found)
    vid.release()
    cv2.destroyAllWindows()
if __name__ = "__main__":
    main()