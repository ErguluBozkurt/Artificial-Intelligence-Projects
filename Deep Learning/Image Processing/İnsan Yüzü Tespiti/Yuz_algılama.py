import cv2
import matplotlib.pyplot as plt

##### İNSAN YÜZÜ TANIMA PROJESİ

# resim
img = cv2.imread("people_face.jpg", 0) 
plt.title("Kişiler")
plt.imshow(img, cmap="gray")
plt.show() 

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_rect = face_cascade.detectMultiScale(img, minNeighbors=3) # minNeighbors başka nesneyi yüz olarak algıladığında değeri arttırarak bu durumdan kurtuluruz
for (x,y,w,h) in face_rect:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 10)
    
plt.title("Kişilerin Yüzü")
plt.imshow(img, cmap="gray")
plt.show() 

# video
cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    if(success):
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors=10)
        
        for (x,y,w,h) in face_rect:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 10)
        cv2.imshow("Kamera", frame)

    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break
    
cap.release()
cv2.destroyAllWindows()
