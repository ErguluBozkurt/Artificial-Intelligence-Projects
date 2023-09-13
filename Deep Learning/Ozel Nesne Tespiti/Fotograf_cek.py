import cv2
import time
import os

path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)# kendi kameramız ile kaydettiğimiz resimler burada saklanacak
# resim boyutu
imgWİdth = 180
imgHeight = 120

# video boyutu
cap = cv2.VideoCapture(0) # kamera tanımlandı
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 180) # kamera aydınlık seviyesi

# negatif ve pozitif resimler için klasör oluşturulur.
while True:
    if not "p" in os.listdir():
        os.makedirs("p")
    elif not "n" in os.listdir():
        os.makedirs("n")
    else:
        print("Klasörler zaten var")
        break

choose = "-"
countSave = 0

while True:
    success, frame = cap.read()
    
    if(success):
        time.sleep(1)
        img = cv2.resize(frame, (imgWİdth,imgHeight)) # resim boyutu ayarlandı
        if(choose == "-"):
            choose = input("Pozitif resim çekmeye başlamak için p harfine basınız. \n=")
            
        if(countSave <= 10 and (choose == "p" or choose == "P")): # 10 pozitif resimi çek kaydet
            cv2.imwrite("p/" + str(countSave) + ".png", img)
            countSave = countSave + 1
            print(countSave)
            
        if(countSave == 11):
            choose = input("Negatif resim çekmeye başlamak için n harfine basınız. \n=")
                
        if(countSave <=20 and (choose == "n" or choose == "N")): # negatif resimleri çek kaydet
            cv2.imwrite("n/" + str(countSave) + ".png", img)
            countSave = countSave + 1
            print(countSave)
            
        if(countSave == 21):
            print("Resim çekme işlemi tamamlandı")
            break
        
        cv2.imshow("Resim", img)
    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break
    
cap.release()
cv2.destroyAllWindows()
