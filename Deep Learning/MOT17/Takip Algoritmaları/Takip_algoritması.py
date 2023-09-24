import cv2
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



##### TAKİP ALGORİTMALARI
"""
1- BOOSTING Takip Algoritması
2- TLD(Tracking, Learning, Detection) Takip Algoritması
3- MEDIANFLOW Takip Algoritması
4- MOSSE(Minimum Output Sum of Squared Error) Takip Algoritması
5- CSRT Takip Algoritması
6- KCF(Kernelized Correlation Filters) Takip Algoritması
7- MIL(Multiple Instance Learning) Takip Algoritması
"""

# Algoritmalar
opencv_object_trackers = {"mil"        : cv2.TrackerMIL_create}



tracker_name = "mil" 
tracker = opencv_object_trackers[tracker_name]()
print("Tracker", tracker_name)

gt = pd.read_csv("Keşifsel Veri Analizi/new_gt.txt")
video_path = "Keşifsel Veri Analizi/MOT17-13-SDP-raw.webm"
cap = cv2.VideoCapture(video_path)

# genel parametreler
fps = 25
frame_number = list()
f = 0
track_list = list()
start_time = time.time() # zaman başlatır kronamometre gibi

while True:
    
    time.sleep(1/fps) # videoyu yavaşlatmak için
    
    success, frame = cap.read()
    if(success):
        frame = cv2.resize(frame, dsize=(960,540))
        (H, W) = frame.shape[:2]
        
        # gt
        car_gt = gt[gt.frame_number == f]
        if(len(car_gt) != 0):
            x = car_gt.x.values[0] 
            y = car_gt.y.values[0] 
            w = car_gt.w.values[0] 
            h = car_gt.h.values[0] 

            center_x = car_gt.center_x.values[0] 
            center_y = car_gt.center_y.values[0] 

            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.circle(frame, (center_x, center_y), 2, (0,0,255), -1)
            track_center_x = int(x+w/2)
            track_center_y = int(y+h/2)
            track_list.append([f,track_center_x,track_center_y])
            info = [("Tracker : ", tracker_name), ("Success : ", "Yes" if success else "No")]
            for (i,(o,p)) in enumerate(info):
                text = f"{o} : {p}"
                cv2.putText(frame, text, (10, H - (i*20) - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0,0,255), 2)
                
        # frame    
        frame_number.append(f)
        f = f + 1
        
        cv2.putText(frame, f"Frame Number:{f}", (10,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
        cv2.imshow("Video", frame)
          
        if(cv2.waitKey(1) & 0xFF == ord("q")): 
            break
        
    else:
        print("Frame Okuma Hatası")
        break
    
cap.release()
cv2.destroyAllWindows()    


# başarı değerlendirmesi
stop_time = time.time()
time_diff = stop_time - start_time

track_df = pd.DataFrame(track_list, columns=["frame_number", "center_x", "center_y"])
if(len(track_df) != 0):
    print(f"Track Method : {tracker}")
    print(f"Time : {time_diff}")
    print(f"Number of frame to track(gt) : {len(gt)}")
    
    track_df_frame = track_df.frame_number
    gt_center_x = gt.center_x[track_df_frame].values
    gt_center_y = gt.center_y[track_df_frame].values

    track_df_center_x = track_df.center_x.values
    track_df_center_y = track_df.center_y.values
    
    plt.plot(np.sqrt((gt_center_x - track_df_center_x)**2 + (gt_center_y - track_df_center_y)**2))
    plt.xlabel("Frame")
    plt.ylabel("Öklid Mesafesi")
    error = np.sum(np.sqrt((gt_center_x - track_df_center_x)**2 + (gt_center_y - track_df_center_y)**2))
    print(f"Toplam Hata : {error}")

