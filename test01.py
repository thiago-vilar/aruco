import cv2
from object_detector import *
import numpy as np
#Load Aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

#Load detector
detector = HomogeneousBgDetector()

cap = cv2.VideoCapture(0)

_, img = cap.read()

#Load_image
img = cv2.imread("phone_aruco.jpg")

#Get aruco marker
corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict , parameters=parameters)

#draw polygon marker
int_corners= np.int0(corners)
# print(int_corners)
cv2.polylines(img, int_corners, True, (0,255, 0), 5)

aruco_perimeter = cv2.arcLength(corners[0], True)

#Pixel to CM ratio

pixel_cm_ratio = aruco_perimeter /20

contours = detector.detect_objects(img)
for cnt in contours:
    #Get rect
    rect = cv2.minAreaRect(cnt)
    (x,y), (w,h), angle = rect

    #Get width and height
    object_width = w /pixel_cm_ratio
    object_height = h / pixel_cm_ratio

   
    #Display rectangle
    box = cv2.boxPoints(rect)
    box=np.int0(box)

    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.polylines(img, [box], True, (255,0,255), 2)
    cv2.putText(img, "Width {}".format(round(w, 1)), (int(x -10), int(y- 15)), cv2.FONT_HERSHEY_PLAIN, 1, (100, 222,0), 2)
    cv2.putText(img, "Height {}".format(round(h, 1)), (int(x -10), int(y +15)), cv2.FONT_HERSHEY_PLAIN, 1, (100, 222,0), 2)


    print(box)

    print(x, y)
    print(w, h)
    print(angle)

cv2.imshow("Image", img)
cv2.waitKey(0)
