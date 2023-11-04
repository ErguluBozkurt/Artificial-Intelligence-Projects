import cv2
import numpy as np
from yolo_model import YOLO

##### YOLO EVRİŞİMSEL SİNİR AĞLARI SINIFLANDIRICISI ile NESNE Sınıflandırma
"""
YOLO'yu bir nesne dedektörü olarak tanımlayabiliriz. Nesne algılama için evrişimli sinir ağlarını kullanır.
Bu projede yolo ile bir köpek ile kedinin fotoğraf üzerinde sınıflandırılması ele alınmıştır.
Not : yolo_model.py içerisindeki dosya yolunu değiştirmeyi unutmayın.
"""

yolo = YOLO(0.6,0.5)
file = "data\coco_classes.txt"
with open(file) as f:
    class_name = f.readlines()
    
all_classes = [c.strip() for c in class_name] # isimlerin sonunda \n vardı, kaldırıldı

# köpeğin resmi images klasörünün içerisinde
path = "images/dog_cat.jpg"
image = cv2.imread(path)
cv2.imshow("Image", image)
if cv2.waitKey(0)== ord("q"):
    cv2.destroyAllWindows() 

pimage = cv2.resize(image, (416,416))
pimage = np.array(pimage, dtype="float32")
pimage /= 255.0
pimage = np.expand_dims(pimage, axis=0)

boxes, classes, scores = yolo.predict(pimage, image.shape)

for box, score, cl in zip(boxes, scores, classes):
    x,y,w,h = box
    
    top = max(0, np.floor(x + 0.5).astype(int))
    left = max(0, np.floor(y + 0.5).astype(int))
    right = max(0, np.floor(x + w + 0.5).astype(int))
    bottom = max(0, np.floor(y + h + 0.5).astype(int)) 
    
    cv2.rectangle(image, (top, left), (right, bottom), (255,0,0),2)
    cv2.putText(image, f"{all_classes[cl]} {score}", (top,left - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0,0,255), 1, cv2.LINE_AA) 
cv2.imshow("Image", image)
if cv2.waitKey(0)== ord("q"):
    cv2.destroyAllWindows() 
