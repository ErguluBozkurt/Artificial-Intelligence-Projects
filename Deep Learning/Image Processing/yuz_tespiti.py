"""
Yüzünüzün size ait olup olmadığını tespiti için bu kodu kullanabilirsiniz.
"""

import cv2
import numpy as np

taniyici = cv2.face.LBPHFaceRecognizer_create()
taniyici.read('kaydet/kaydet.yml')  #okunacak dosya tanımlandı
yolsiniflandirici = "haarcascade_frontalface_default.xml"
yuzsiniflandirici = cv2.CascadeClassifier(yolsiniflandirici)
font = cv2.FONT_HERSHEY_SIMPLEX  #yazı tipi tanımlandı
vid_cam = cv2.VideoCapture(0)   # kamera tanımlandı

while True:
    
    ret, kamera = vid_cam.read()  #kamera okutuldu
    gri = cv2.cvtColor(kamera,cv2.COLOR_BGR2GRAY)
    yuzler = yuzsiniflandirici.detectMultiScale(gri, 1.2,5)
    
    for(x,y,w,h) in yuzler:
        
        cv2.rectangle(kamera, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)  
        Id,conf = taniyici.predict(gri[y:y+h,x:x+w])
        print(Id)
        
        if(Id == 1):
            Id = "Ben"
        
             
        # elif(Id == 2)
        #     Id = isim  şeklinde kişi ekleyebiliriz
            
        cv2.rectangle(kamera, (x-22, y-90), (x+w+22, y-22), (0,255,0), -1)  
        cv2.putText(kamera, str(Id), (x,y-40), font, 2, (255,255,255), 3)   #yazılacak isimin ebatları belirlendi
        
    cv2.imshow('kamera', kamera)  #kamera göster komutu
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
vid_cam.release()  #kamera durduruldu
cv2.destroyAllWindows()
    
            