import cv2
import numpy as np
from ultralytics import YOLO
import imutils 

"""
Bu projede otoyol dan geçen araçların takibi ve toplam sayısını elde ettik.
"""

video_path = "inference/test.mp4"
model_path = "models/yolov8n.pt" 

cap = cv2.VideoCapture(video_path)
model = YOLO(model_path)

# tespit edilecek araçların id numaraları
vehicle_ids = [2,3,5,7]

n = 0 # araç sayısı
while True:
    success, frame = cap.read()
    
    if(success):
        frame = imutils.resize(frame, width=1280) 
        
        results = model.track(frame,  persist = True, verbose=False)[0] 
        
        # track_ids = results.boxes.id.int().cpu.tolist() # takip id değerleri
        bboxes = np.array(results.boxes.data.tolist(), dtype="int") 
        
        cv2.line(frame, (0,430), (1280,430), (0,0,255), 2) # referans çizgisi çekelim. araç bu çizgiyi geçtiğinde sayılacak
        
        for box in bboxes:
            x1, y1, x2, y2, track_id, score, class_id = box
            if class_id in vehicle_ids:
                class_name = results.names[int(class_id)].upper() 

                text = f"ID : {track_id} {class_name}"
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255, 0), 2)
                
                
                center_y = y2/2
                # print(center_y)
                if(center_y > 229 and center_y < 231):
                    n += 1
                
                cv2.putText(frame, f"Total Vehicle : {n}", (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0, 255), 2)
        
        cv2.imshow("Frame", frame)
        if(cv2.waitKey(1) & 0xFF == ord("q")):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()
