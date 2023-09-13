import cv2
import os


##### KEDİ YÜZÜ TANIMA PROJESİ

files = os.listdir() # dizinde bulunan klasörleri listeler
print(files)

# Resimleri listeye ekle
img_path_list = list()
for f in files:
    if(f.startswith("kedi")):
        img_path_list.append(f)
print(img_path_list)

# Resimleri içe aktar ve görselleştir
for j in img_path_list:
    print(j)
    img = cv2.imread(j) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
    
    rect = detector.detectMultiScale(gray, scaleFactor=1.0054, minNeighbors=4) # scaleFactor ne kadar zoom yapacağını söyler

    for (i, (x,y,w,h)) in enumerate(rect):
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 2)
        cv2.putText(img, f"Kedi {i+1}", (x, y-10),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
    
    cv2.imshow(f"Kedi {j}", img) 
    if(cv2.waitKey(0) & 0xFF == ord("q")):
        continue
