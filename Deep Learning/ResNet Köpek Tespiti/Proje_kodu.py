import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input 
from tensorflow.keras.preprocessing.image import img_to_array 
from keras.applications import imagenet_utils


##### ResNet EVRİŞİMSEL SİNİR AĞLARI SINIFLANDIRICISI ile NESNE TESPİTİ
# Piramit Gösterimi
def image_pyramid(image, scale=1.5, minSize=(224,224)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = cv2.resize(image, dsize=(w,w))
        if(image.shape[0] < minSize[1] or image.shape[1] < minSize[0]):
            break
        yield image

# Kayan Pencere 
def sliding_window(image, step, ws):
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            yield(x, y, image[y:y+ws[1], x:x+ws[0]])
            

# Maksimum Olmayan Bastırma 
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


WIDTH = 600 # resmin genişliği
HEIGHT = 600 # resmin yüksekliği
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (200,150)
INPUT_SIZE = (224,224)

print("Resnet Yükleniyor")
model = ResNet50(weights = "imagenet", include_top = True)

orig = cv2.imread("sibirya-kurdu-02.jpg")
orig = cv2.resize(orig, dsize=(WIDTH, HEIGHT))
(H, W) = orig.shape[:2]

# resim piramidi
pyramid = image_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE)

rois = list()
locs = list()

for image in pyramid:
    scale = W / float(image.shape[1])
    
    for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
        x = int(x * scale)
        y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)

        roi = cv2.resize(roiOrig, INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        
        rois.append(roi)
        locs.append((x, y, x+w, y+h))
        
rois = np.array(rois, dtype="float32")

print("Sınıflandırma İşlemi")
preds = model.predict(rois)

preds = imagenet_utils.decode_predictions(preds, top=1)
labels = {}
min_conf = 0.9 # başarı %90 olanları sadece göster

for (i, p) in enumerate(preds):
    (_, label, prob) = p[0]
    if(prob >= min_conf):
        box = locs[i]
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L
        
for label in labels.keys():
    clone = orig.copy()
    for (box, prob) in labels[label]:
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0,255,0), 2)
        
    cv2.imshow("Maxima Uygulanmadı", clone)
    if(cv2.waitKey(0) & 0xFF == ord("q")):
        cv2.destroyAllWindows() 
    
    # maxima uygula
    clone = orig.copy()
    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes, proba)
    
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0,255,0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.45, (0,255,0), 2)
    
    cv2.imshow("Maxima Uygulandı", clone)
    
    if(cv2.waitKey(0) & 0xFF == ord("q")):
        cv2.destroyAllWindows() 
