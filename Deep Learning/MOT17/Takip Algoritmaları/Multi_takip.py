import cv2
import time


##### ÇOKLU NESNE TAKİBİ
# Veri seti kaynağı : https://motchallenge.net/vis/MOT17-04-SDP

# Algoritmalar
opencv_object_trackers = {"mil" : cv2.TrackerMIL_create}

tracker_name = "mil" 
trackers = cv2.MultiTracker_create()

video_path = "Takip Algoritmaları/MOT17-04-SDP-raw.webm"
cap = cv2.VideoCapture(video_path)

# genel parametreler
fps = 30
f = 0


while True:
    
    time.sleep(1/fps) # videoyu yavaşlatmak için
    
    success, frame = cap.read()
    if(success):
        frame = cv2.resize(frame, dsize=(960,540))
        (H, W) = frame.shape[:2]
        
        (success2, boxes) = trackers.update(frame)
        info = [("Tracker", tracker_name), ("Success", "Yes" if success2 else "No")]
        for (i,(o,p)) in enumerate(info):
            text = f"{o} : {p}"
            string_text = string_text + text + " "
        cv2.putText(frame, string_text, (10,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0,0,255), 2)

        for box in boxes:
            (x,y,w,h) = [int(i) for i in box]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,0), 2)

        cv2.imshow("Video", frame)
        key = cv2.waitKey(1) & 0xFF         
        if(key == ord("w")): # manuel seçme işlemi. 
            box = cv2.selectROI("Video", frame, fromCenter = False)
            tracker = opencv_object_trackers[tracker_name]()
            trackers.add(tracker, frame, box)
        elif(key == ord("q")):
            break

cap.release()
cv2.destroyAllWindows()    
