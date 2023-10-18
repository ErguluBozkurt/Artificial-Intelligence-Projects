import cv2
import numpy as np
from keras.applications import imagenet_utils
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input 
from tensorflow.keras.preprocessing.image import img_to_array


##### ResNet EVRİŞİMSEL SİNİR AĞLARI SINIFLANDIRICISI ile NESNE TESPİTİ
# Bu projede seçmeli arama nesne tespit yöntemi uygulandı.

"""
Seçmeli arama, süper piksel algoritması kullanarak bir görseli aşırı bölümlere ayırma yöntemidir.
Süper piksel, ortak özellikleri(piksel yoğunluğu gibi) paylaşan bir piksel grubu olarak tanımlanabilir.
Seçmeli arama, beş temel benzerlik ölçsüne dayalı olarak süper pikselleri hiyerarşik bir şekilde birleştirir:
    Renk Benzerliği
    Doku Benzerliği
    Boyut Benzerliği
    Şekil Benzerliği
    Yukarıdaki benzerliklerin doğrusal kombinasyonu

"""
def selective_search(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality()

    print("Başladı")
    rects = ss.process()

    return(rects[:1000]) # eğitim kısa sürsün diye ilk 1000 tanesini aldık




def non_max_suppression(boxes, probs = None, overlapThresh = 0.3):
    if(len(boxes) == 0):
        return([])
    if(boxes.dtype.kind == "i"):
        boxes = boxes.astype("float")
    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # alanı bulalım
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2
    
    # olasılık değerleri
    if probs is not None:
        idxs = probs
    
    # indeksi sırala
    idxs = np.argsort(idxs)
    pick = list() # seçilen kutular
    
    while(len(idxs) > 0):
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # x ve y nin en büyük ve en küçük değerleri
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.maximum(x2[i], x2[idxs[:last]])
        yy2 = np.maximum(y2[i], y2[idxs[:last]])

        # w, h bulalım
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # overlap
        overlap = (w*h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        
    return(boxes[pick].astype("int"))


# model
print("Resnet Yükleniyor")
model = ResNet50(weights = "imagenet", include_top = True)
image = cv2.imread("animals.png")
image = cv2.resize(image, dsize=(300, 300))
(H, W) = image.shape[:2]

rects = selective_search(image)
proposals = list()
boxes = list()

for (x,y,w,h) in rects:
    if(w / float(W) < 0.1 or h / float(H) < 0.1):
        continue
    
    roi = image[y:y+h, x:x+w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (224,224))
    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    proposals.append(roi)
    boxes.append((x,y,w,h))

proposals = np.array(proposals)

print("Sınıflandırma İşlemi")
preds = model.predict(proposals)
preds = imagenet_utils.decode_predictions(preds, top=1)

labels = {}
min_conf = 0.9 # başarı %90 olanları sadece göster

for (i, p) in enumerate(preds):
    (_, label, prob) = p[0]
    if(prob >= min_conf):
        (x,y,w,h) = boxes[i]
        box = (x,y,x+w, y+h)
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

clone = image.copy()
for label in labels.keys():
    for (box, prob) in labels[label]:
        boxes = np.array([p[0] for p in labels[label]])
        proba = np.array([p[1] for p in labels[label]])
        boxes = non_max_suppression(boxes, proba)

        for (startX, startY, endX, endY) in boxes:
            cv2.rectangle(clone, (startX, startY), (endX, endY), (0,255,0), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0,255,0), 2)
        
    cv2.imshow("Maxima Uygulandı", clone)
    
    if(cv2.waitKey(0) & 0xFF == ord("q")):
        cv2.destroyAllWindows() 
