import os
import cv2
import numpy as np
import darknet
import time
from ctypes import *

class droneDetector:
    def __init__(self, configPath, weightPath, metaPath):
        self.configPath = configPath
        self.weightPath = weightPath
        self.metaPath = metaPath
        self.netMain = None
        self.metaMain = None
        self.altNames = None
        self.darknet_image = darknet.make_image(640, 480,3)

    def initYOLO(self):
        if not os.path.exists(self.configPath):
            raise ValueError("Invalid config path `" +
                            os.path.abspath(self.configPath)+"`")
        if not os.path.exists(self.weightPath):
            raise ValueError("Invalid weight path `" +
                            os.path.abspath(self.weightPath)+"`")
        if not os.path.exists(self.metaPath):
            raise ValueError("Invalid data file path `" +
                            os.path.abspath(self.metaPath)+"`")
        if self.netMain is None:
            self.netMain = darknet.load_net_custom(self.configPath.encode(
                "ascii"), self.weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if self.metaMain is None:
            self.metaMain = darknet.load_meta(self.metaPath.encode("ascii"))
        if self.altNames is None:
            try:
                with open(self.metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                    re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

    def convertBack(self, x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        
        return xmin, ymin, xmax, ymax


    def cvDrawBoxes(self, detections, img):
        for detection in detections:
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = self.convertBack(
                float(x), float(y), float(w), float(h))
            if detection[1] > 0.3:
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymax)
                cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(img,
                            detection[0].decode() +
                            " [" + str(round(detection[1] * 100, 2)) + "]",
                            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            [0, 255, 0], 2)
        
        return img

    def drone_detect(self, frame):
        prev_time = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        darknet.copy_image_from_bytes(self.darknet_image,frame_rgb.tobytes())
        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh=0.25)
        print((time.time()-prev_time))

        return detections
    
    def drawDrone(self, frame, detections):
        image = self.cvDrawBoxes(detections, frame)
        # cv2.imshow('Demo', image)
        # cv2.waitKey(1)

        return image
            
            
if __name__ == "__main__":
    configPath = '/media/harddisk/TCS_drone_data/Drones/yuen/drone_for_yolo/yolov3-tiny.cfg'
    weightPath = '/media/harddisk/TCS_drone_data/Drones/yuen/drone_for_yolo/backup_2/yolov3-tiny_best.weights'
    metaPath = '/media/harddisk/TCS_drone_data/Drones/yuen/drone_for_yolo/drone.data'
    drone_detector = droneDetector(configPath, weightPath, metaPath)
    drone_detector.initYOLO()
    frame = cv2.imread('/media/harddisk/TCS_drone_data/Drones/usc drone/Drone8/img/0087.jpg')
    frame = cv2.resize(frame, (640, 480))
    detections = drone_detector.drone_detect(frame)
    out = drone_detector.drawDrone(frame, detections)
    cv2.imshow('Out', out)
    cv2.waitKey()