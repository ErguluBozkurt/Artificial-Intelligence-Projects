import cv2
import numpy as np
import pickle


## MODELİN ÇALIŞMASI
# Bu kod daha önceden eğitilmiş modelin uygulanması için hazırlanan bir koddur.

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return(img)

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 480)

pickle_in = open("model_trained_new.p", "rb")
model = pickle.load(pickle_in)

while True:
    success, frame = cap.read()
    img = np.asarray(frame)
    img = cv2.resize(img, (32,32))
    img = preProcess(img)
    
    img = img.reshape(1,32,32,1)
    
    # tahmin işlemi
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    propVal = np.amax(predictions)
    print(classIndex, propVal)
    
    if(propVal > 0.7):
        cv2.putText(frame, str(classIndex) + "  " + str(propVal), (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)
    
    cv2.imshow("Rakam Sınıflanfırma", frame)
    
    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break
