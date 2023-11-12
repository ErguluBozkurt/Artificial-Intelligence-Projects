import cv2
import imutils
from ultralytics import YOLO
import os

"""
Bu projede yolov8 kullanılarak beyin tümörünü tespit eden bir eğitim modeli yapılmıştır.
Eğitim sonucunda görseller üzerinde tümör tespit işlemini başarılı şekilde sonuçlandırmıştır.
"""

files = os.listdir() # dizinde bulunan klasörleri listeler
print(files)

img_path_list = list()
for f in files:
    if(f.endswith("jpg")):
        img_path_list.append(f)
print(img_path_list) # resim yolu

model_path = "best.pt"

for img_path in img_path_list:
    
    model = YOLO(model_path)
    img = cv2.imread(img_path)
    img = imutils.resize(img, width=480)

    results = model(img)[0] # model sonuçları

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)
        if score > 0.8:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_name = results.names[class_id]
            score = score * 100
            text = class_name + ":" + str(round(score, 2)) + "%"
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 255, 255), 1)

    cv2.imshow("Image", img)
    if(cv2.waitKey(0) & 0xFF == ord("q")):
        continue