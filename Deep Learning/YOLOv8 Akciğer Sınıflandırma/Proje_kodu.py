import cv2
import numpy as np
from ultralytics import YOLO
import os
"""
Bu projede yolov8 kullanılarak covid, pneumoina ve normal bir insanın sahip olduğu akciğer görselleri üzerinde eğitim yapılmış ve 
yüksek doğruluk elde edilmiştir. 


"""

files = os.listdir() # dizinde bulunan klasörleri listeler
print(files)

img_path_list = list()
for f in files:
    if(f.endswith("3.png")):
        img_path_list.append(f)
print(img_path_list) # resim yolu

model_path = "best.pt" # best yolu

for img in img_path_list: 
    model = YOLO(model_path)
    result = model(img)

    class_dict = result[0].names # sınıfı
    probs = result[0].probs.data.tolist() # başarı olasılığı

    name = class_dict[np.argmax(probs)] # np.argmax(probs) en büyük olasılık değerine sahip indexi verir
    max_prob = np.max(probs)
    text = "Result : " + name + "  Score : " + str(round(max_prob, 2)) + "%"

    img = cv2.imread(img)
    cv2.putText(img, text, (10,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.55, (0,0,255), 1)
    cv2.imshow("Resim",img)
    
    if(cv2.waitKey(0) & 0xFF == ord("q")):
        continue