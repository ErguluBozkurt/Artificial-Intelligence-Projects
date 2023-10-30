import cv2
import numpy as np

##### SSD ile Nesne Tespiti
"""
SSD, kayan pencere kullanmak yerine görüntüyü bir ızgara kullanarak böler ve her bir ızgara hücresinin görüntünün o bölgesindeki
nesneleri tespit etmekten sorumlu olmasını sağlar.
"""

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0,255,size=(len(CLASSES),3))

proto_url = "MobileNetSSD_deploy.prototxt.txt"
caffe_url = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto_url, caffe_url)

cap = cv2.VideoCapture(0) # ismini degistir
cap.set(3,800)
cap.set(4,600)
        
while True:
    success, frame = cap.read()
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300, 300)), 0.007843,(300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    for j in np.arange(0, detections.shape[2]):
        confidence = detections[0,0,j,2]
        
        if confidence > 0.10:
            idx = int(detections[0,0,j,1])
            box = detections[0,0,j,3:7]*np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            
            label = f"{CLASSES[idx]} : {round(float(confidence),2)}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx],2)
            y = startY - 16 if startY -16 >15 else startY + 16
            cv2.putText(frame, label, (startX,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5,COLORS[idx],2)
            
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
