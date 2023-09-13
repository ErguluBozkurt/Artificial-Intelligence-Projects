import cv2
import numpy as np
from collections import deque
"""
Bu projede amaç mavi renkli cisimleri tespit etmektir.
"""

buffer_size = 16 # deque nun boyutu 
pts = deque(maxlen=buffer_size)

blue_lower = (90, 110, 0) # (H,S,V) Rengimiz mavi.
blue_upper = (165, 255, 255)

# Kamerayı aç
cap = cv2.VideoCapture(0)
cap.set(3, 960) # Kamera genişliği 960
cap.set(4, 480) # Kamera yüksekliği 480

while True:
    success, frame = cap.read() 
    if(success): 
        # blur
        blured = cv2.GaussianBlur(frame, (11,11), 0)
        # HSV
        hsv = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV Image", hsv)

        # mavi için maske oluştur
        mask = cv2.inRange(hsv, blue_lower, blue_upper)
        cv2.imshow("Mask Image", mask)

        # Gürültü var azaltmak için erozyon ve genişleme kullan
        mask = cv2.erode(mask, None, iterations = 2)
        mask = cv2.dilate(mask, None, iterations = 2)
        cv2.imshow("Mask + Erozyon + Genişleme Image", mask)

        # Kontur
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None
        if(len(contours) > 0):
            c = max(contours, key=cv2.contourArea) 
            rect = cv2.minAreaRect(c) # diktörtgen ile çevir
            ((x,y), (width, height), rotation) = rect

            box = cv2.boxPoints(rect) # kutucuk yapalım
            box = np.int64(box)
            
            mom = cv2.moments(c) # görüntünün merkezini bulur
            center = (int(mom["m10"] / mom["m00"]), int(mom["m01"] / mom["m00"]))

            s = f"x : {np.round(x)} y : {np.round(y)} width : {np.round(width)} height : {np.round(height)} rotation : {np.round(rotation)}"

            cv2.drawContours(frame, [box], 0,( 0,255,255), 2)# kontoru çizdir.
            cv2.circle(frame, center, 5, (255,0,255), -1) # merkeze bir nokta koyalım. 
            cv2.putText(frame, s, (20,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0,0,0), 1) 

        # çizgi ile takip algoritması
        pts.appendleft(center)
        for i in range(1, len(pts)): # çizginin uzunluğu tanımlanan deque ile değişmektedir.
            if(pts[i-1] is None or pts[i] is None):
                continue
            cv2.line(frame, pts[i-1], pts[i], (0,255,0), 3) # yeşil çizgi

        cv2.imshow("Orjinal Tespit", frame)

    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break

