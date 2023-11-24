import cv2
import numpy as np
import imutils
from ultralytics import YOLO

img_path = "vucut.jpg"
model_path = "best.pt"

img = cv2.imread(img_path)
img = imutils.resize(img, width=640)

model = YOLO(model_path)
results = model(img)[0]

for result in results:
  points = np.array(result.keypoints.xy.cpu(), dtype="int")

  for point in points:
    for p in point:
      cv2.circle(img, (p[0], p[1]), 3, (0,255,0), -1)
cv2.imshow("Image",img)
if(cv2.waitKey(0) & 0xFF == ord("q")):
    cv2.destroyAllWindows()