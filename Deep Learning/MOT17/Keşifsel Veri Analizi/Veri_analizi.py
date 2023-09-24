import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

##### KEŞİFSEL VERİ ANALİZİ

# 1. Adım
# Veri seti kaynağı : https://motchallenge.net/vis/MOT17-13-SDP
# Makalede : https://arxiv.org/pdf/1603.00831.pdf


# 2. Adım
path_in = "Keşifsel Veri Analizi/img1" 
path_out = "Keşifsel Veri Analizi/deneme.mp4" 

files = [f for f in os.listdir(path_in) if os.path.isfile(os.path.join(path_in, f))]
img = cv2.imread(path_in + "\\" + files[44]) # 44. resime bakalım
img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
plt.imshow(img)

# Resimleri videoya dönüştür.
fps = 25 
size = (1920,1080) 
out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*"MP4V"), fps, size, True)

for i in files:
    print(i)
    file_name = path_in + "\\" + i
    img = cv2.imread(file_name)
    out.write(img)
    
out.release()

# 3. Adım
col_list = ["frame_number", "identity_number", "left", "top", "width", "height", "score", "class", "visibility"] # gt sutunları
data = pd.read_csv("Keşifsel Veri Analizi/gt.txt", names = col_list)
print(data.head())

sns.countplot(data["class"])
plt.show()

car = data[data["class"] == 3] # araçların classı 3, insanlarınki farklı hangisini almak istersek o yazılır. Makalede yazıyor.
video_path = "Keşifsel Veri Analizi/MOT17-13-SDP-raw.webm"
cap = cv2.VideoCapture(video_path)
id = 29 # kutu numarası 29 olan aracı takip edecez
number_image = np.max(data["frame_number"])
fps = 25
bound_box_list = list() # kutucukları depolamak için

for i in range(number_image - 1):
    success, frame = cap.read()
    time.sleep(0.05)
    if(success):
        frame = cv2.resize(frame, dsize=(960,540))
        
        filter_id = np.logical_and(car["frame_number"] == i + 1, car["identity_number"] == id)
        
        if(len(car[filter_id]) != 0):
            x = int(car[filter_id].left.values[0] / 2)
            y = int(car[filter_id].top.values[0] / 2)
            w = int(car[filter_id].width.values[0] / 2)
            h = int(car[filter_id].height.values[0] / 2)

            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.circle(frame, (int(x+w/2), int(y+h/2)), 2, (0,0,255), -1)
            # frame, x, y, genislik, yukseklik, center x, center y 
            bound_box_list.append([i, x, y, w, h, int(x+w/2), int(y+h/2)])
            
        cv2.putText(frame, f"frame num:{i+1}", (10,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
        cv2.imshow("Video", frame)
        
        if(cv2.waitKey(1) & 0xFF == ord("q")):
            break
    else:
        print("Frame Okunmadı")
        break
cap.release()
cv2.destroyAllWindows()

# 4. Adım
df = pd.DataFrame(bound_box_list, columns=["frame_number", "x", "y", "w", "h", "center_x", "center_y"])
df.to_csv("Keşifsel Veri Analizi/new_gt.txt", index=False)