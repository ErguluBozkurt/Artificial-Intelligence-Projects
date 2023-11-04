import cv2
import numpy as np
from yolo_model import YOLO

"""
YOLO'yu bir nesne dedektörü olarak tanımlayabiliriz. Nesne algılama için evrişimli sinir ağlarını kullanır.
Bu projede yolo ile bir köpek ile kedinin kamera üzerinde sınıflandırılması ele alınmıştır.
Not : yolo_model.py içerisindeki dosya yolunu değiştirmeyi unutmayın.
"""

yolo = YOLO(0.6,0.5)
file = "data\coco_classes.txt"
with open(file) as f:
    class_name = f.readlines()
    
all_classes = [c.strip() for c in class_name] 


cap = cv2.VideoCapture(0) # Kamera

while True:
    _, frame = cap.read()
    pimage = cv2.resize(frame, (416,416))
    pimage = np.array(pimage, dtype="float32")
    pimage /= 255.0
    pimage = np.expand_dims(pimage, axis=0)

    boxes, classes, scores = yolo.predict(pimage, frame.shape)
    if(classes != None): # Kamera hemen açılmadığı için hata engellendi
        for box, score, cl in zip(boxes, scores, classes):
            
            x,y,w,h = box
            
            top = max(0, np.floor(x + 0.5).astype(int))
            left = max(0, np.floor(y + 0.5).astype(int))
            right = max(0, np.floor(x + w + 0.5).astype(int))
            bottom = max(0, np.floor(y + h + 0.5).astype(int)) 
            
            cv2.rectangle(frame, (top, left), (right, bottom), (255,0,0),2)
            cv2.putText(frame, f"{all_classes[cl]} {score}", (top,left - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow("Frame", frame)
        if(cv2.waitKey(1) & 0xFF == ord("q")):
            break

cap.release()
cv2.destroyAllWindows()