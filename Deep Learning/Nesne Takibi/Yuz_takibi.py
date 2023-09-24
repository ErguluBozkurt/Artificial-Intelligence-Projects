import cv2

##### NESNE TAKİP PROJESİ
# Ortalama Kayma Algoritması (Meanshift)

# Kamera aç
cap = cv2.VideoCapture(0)
n = 0
if not cap.isOpened():
    print("Kamera açma hatası")
else:
    
    while True:
        success, frame = cap.read() # frame oku
        if not success:
            print("Kare okuma hatası")
            break

        
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # detection
        face_rect = face_cascade.detectMultiScale(frame)

        if len(face_rect) > 0:
            (x, y, w, h) = tuple(face_rect[0])
            track_window = (x, y, w, h) # meanshift algoritması girdisi
                        
            # regions of interest
            roi = frame[y:y+h, x:x+w] # roi = face

            hsv_rio = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            roi_hist = cv2.calcHist([hsv_rio], [0], None, [180], [0,180]) 
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

            # takip için gerekli durdurma kriterleri
            # count : hesaplanacak maksimum oge sayısı
            # eps : değişiklik
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1)
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            x,y,h,w = track_window            
                            
            img = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 5)
            cv2.imshow("Kamera", img)   
        else:
            n = n+ 1
            print(f"Nesne Tespit Edilemedi Tekrar Deneniyor...{n}")
            
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
