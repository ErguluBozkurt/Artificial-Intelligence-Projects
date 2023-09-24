import cv2
import os


# YAYA TESPİTİ PROJESİ

files = os.listdir()
print(files)
os.chdir('Yaya Tespiti')
files = os.listdir()
print(files)

img_path_list = list()
for i in files:
    if(i.startswith("yaya")):
        img_path_list.append(i)
        
print(img_path_list)

# hog tanımlayıcısı
hog = cv2.HOGDescriptor()
# Tanımlayıcıya SVM ekle
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for img_path in img_path_list:
    img = cv2.imread(img_path)
    (rects, weights) = hog.detectMultiScale(img, padding=(8,8), scale=1.05)
    
    for (x,y,w,h) in rects:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255),2)
    
    cv2.imshow("Yaya", img)
    
    if(cv2.waitKey(0) & 0xFF == ord("q")):
        continue

