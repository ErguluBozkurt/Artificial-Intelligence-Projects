"""
Fotoğraf çekildikten sonra kayıt işlemi için aşağıdaki kod yazılmıştır.
"""

import cv2, os
import numpy as np
from PIL import Image

tanıyıcı =cv2.face.LBPHFaceRecognizer_create()
dedektor = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    yuzornekleri = []  #değişken tanımlandı
    isimler = []  #değişken tanımlandı
    for imagePaths in imagePaths:
        
        PIL_img = Image.open(imagePaths).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePaths)[-1].split(".")[1])
        print(id)
        yuzler = dedektor.detectMultiScale(img_numpy)
        
        for (x,y,w,h) in yuzler:
            
            yuzornekleri.append(img_numpy[y:y+h,x:x+w])
            isimler.append(id)
            
    return yuzornekleri,isimler

yuzler,isimler = getImagesAndLabels('veri')  #resimlerin bulunduğu klasör 
tanıyıcı.train(yuzler, np.array(isimler))
tanıyıcı.save('kaydet/kaydet.yml')  #kaydedilecek klasör belirlendi