"""
Aşağıda bulunan kod ile fotoğrafınızı çekebilirsiniz.
"""
import cv2
vid_cam = cv2.VideoCapture(0)  #kamera tanıtıldı
yuz_dedektor = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
yuz_ismi = 1  # yeni çekilecek kişi için değiştirmememiz gerkli
sayi = 1  # kaç fotoraf çekileceğimimizi saydırmak için 

while(True):
    
    _,resim_cerceve = vid_cam.read()  #kamera okutuldu
    gri = cv2.cvtColor(resim_cerceve, cv2.COLOR_BGR2GRAY)  #resim rengi için gri yapıldı
    yuzler = yuz_dedektor.detectMultiScale(gri, 1.3, 5)
    
    for (x,y,w,h) in yuzler:
        
        cv2.rectangle(resim_cerceve, (x,y), (x+w,y+h), (0, 255, 0), 2)
        sayi += 1
        cv2.imwrite("veri/User." + str(yuz_ismi) + '.' + str(sayi) + ".jpg", gri[y:y+h,x:x+w])
        cv2.imshow('cerceve', resim_cerceve)  #kamerayı açma komutu
        
    if cv2.waitKey(10) & 0xFF == ord('q'):  #fotoğraf kalitesi, kamerayı kaptmak için atandı
        break
    
    elif sayi>1:  # fotograf sayisi sınırlandı
          break
      
vid_cam.release()  #kamera kapandı
cv2.destroyAllWindows()
      