import cv2
import random
import numpy as np
import pickle 
from tensorflow.keras.preprocessing.image import img_to_array 


##### R-CNN EVRİŞİMSEL SİNİR AĞLARI SINIFLANDIRICISI
# Bu projede rakamların sınıflandırılması için model geliştirildi. Model eğitimi daha önceden yapılmıştır.
# Modelin eğitimi hakkında bilgi githup içerisinde yer alıyor.

image = cv2.imread("MNIST.jpg")
cv2.imshow("Image", image)
if cv2.waitKey(0) == ord("q"):
    cv2.destroyAllWindows() 
    
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchQuality()

print("Start")
rects = ss.process()

proposals = list()
boxes = list()
output = image.copy()

for (x,y,w,h) in rects:
    color = [random.randint(0,255) for j in range(0,3)]
    cv2.rectangle(output, (x,y), (x+w, y+h), color, 2)
    
    roi = image[y:y+h, x:x+w]
    roi = cv2.resize(roi, dsize=(32,32), interpolation=cv2.INTER_LANCZOS4)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    roi = img_to_array(roi)
    
    proposals.append(roi)
    boxes.append((x,y,x+w,y+h))

proposals = np.array(proposals, dtype="float32")
boxes = np.array(boxes, dtype="int32")

print("Classification")
pickle_in = open("model_trained_new.p", "rb")
model = pickle.load(pickle_in)
proba = model.predict(proposals)

number_list = list()
idx = list()

for i in range(len(proba)):
    max_prob = np.max(proba[i,:])
    if(max_prob > 0.95): # score % 95 
        idx.append(i)
        number_list.append(np.argmax(proba[i]))
        
for i in range(len(number_list)):
    j = idx[i]
    cv2.rectangle(image, (boxes[j,0], boxes[j,1]), (boxes[j,2], boxes[j,3]), (0,0,255),2)
    cv2.putText(image, str(np.argmax(proba[j])), (boxes[j,0] + 5, boxes[j,1] + 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0,255,0), 1)
    cv2.imshow("Iamge", image)
    if(cv2.waitKey(0) & 0xFF == ord("q")):
       cv2.destroyAllWindows() 
    