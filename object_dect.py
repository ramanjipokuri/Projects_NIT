import cv2
import time
import numpy as np
import imutils

cam = cv2.VideoCapture(1)
time.sleep(1)

firstFrame = None
area = 500

while True:
    _,img = cam.read()
    text = "Normal"
    img = imutils.resize(img, width=500)
    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gaussian_blurred = cv2.GaussianBlur(grayImg, (21, 21), 0)
    
    if firstFrame is None:
        firstFrame = gaussian_blurred
        continue
    imgDiff = cv2.absdiff(firstFrame, gaussian_blurred)
    
    tresh = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
    tresh = cv2.dilate(tresh, None, iterations=2)
    
    cnts = cv2.findContours(tresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)
        text = "Movement Detected"
        
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("cameraFeed", img)
    
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()