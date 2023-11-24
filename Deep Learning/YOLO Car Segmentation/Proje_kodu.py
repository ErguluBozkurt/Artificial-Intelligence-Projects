import numpy as np
import cv2
import imutils
from ultralytics import YOLO


def extract_data(img, model):
  h, w, ch = img.shape
  results = model.predict(source=img.copy(), save=False, save_txt=False)
  result = results[0]
  seg_contour_idx = []

  for seg in result.masks.xy:
    seg[:,0] = seg[:,0] * w
    seg[:,1] = seg[:,1] * h
    segment = np.array(seg, dtype=np.int32)
    seg_contour_idx.append(segment)

  bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int") # result a git, kutuları bul, bu kutuların xyxy kordinatlarını bul ve tensör olarak dönder
  class_ids = np.array(result.boxes.cls.cpu(), dtype="int") # classı bul ve tensör olarak dönder
  scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2) # scoru bul ve tensör olarak dönder
  class_names = result.names

  return(bboxes, class_ids, seg_contour_idx, scores, class_names)


img_path = "car_image.jpeg"
model_path = "best.pt"

img = cv2.imread(img_path)
img = imutils.resize(img, width=480)

model = YOLO(model_path)

bboxes, class_ids, seg_contour_idx, scores, class_names = extract_data(img, model)

for box, class_id, segmentation_id, score in zip(bboxes, class_ids, seg_contour_idx, scores):
  (xmin, ymin, xmax, ymax) = box

  cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

  class_name = class_names[class_id]
  score = score * 100
  text = f"{class_name}: %{round(score, 2)}"
  cv2.putText(img, str(text), (xmin-10,ymin-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0,255,0), 1)

cv2.imshow("Image", img)
if(cv2.waitKey(0) & 0xFF == ord("q")):
    cv2.destroyAllWindows()
