import cv2
"""
Proje sarı kalemi tespit etmesi amacı ile yazılmıştır.
Nesne tespitini doğru şekilde yapmadığı takdirde projeyi yeniden eğitiniz ve deneyiniz.
"""

object_Name = "Sarı Kalem"
frameWidth = 280
frameHeight = 360
color = (255,0,0)

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a): # boş fonksiyon
    pass

# trackbar
cv2.namedWindow("Sonuç")
cv2.resizeWindow("Sonuç", frameWidth, frameHeight + 100)
cv2.createTrackbar("Scale", "Sonuç", 400, 1000, empty)
cv2.createTrackbar("Neighbor", "Sonuç", 4, 50, empty)

# cascade classifier
cascade = cv2.CascadeClassifier("Deep Learning\Codes\Ozel Nesne Tespiti\classifier\cascade.xml")

while True:
    
    success, frame = cap.read()
    
    if(success):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale_val = 1 + (cv2.getTrackbarPos("Scale", "Sonuç") / 1000)
        neighbor = cv2.getTrackbarPos("Neighbor", "Sonuç")

        rect = cascade.detectMultiScale(gray, scale_val, neighbor)
        
        for (x,y,w,h) in rect:
            cv2.rectangle(frame, (x,y), (x + w, y + h), color, 3)
            cv2.putText(frame, object_Name, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
        
        cv2.imshow("Kamera", frame)

    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break
        
