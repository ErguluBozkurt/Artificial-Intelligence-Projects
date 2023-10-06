import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pickle
from sklearn.metrics import confusion_matrix

##### RAKAM SINIFLANDIRMA
# Bu projede rakamların olduğu veri seti hazırlandıktan sonra eğitimin gerçekleşmesi için hazırlanan kod yer alır.

path = "myData" # resimlerin içerdiği klasör ismi
myList = os.listdir(path)
no_classes = len(myList)
print(f"Label sınıf sayısı : {no_classes}")

images = list()
class_no = list()

for i in range(no_classes):
    myImageList = os.listdir(path + "\\" + str(i))
    for j in myImageList:
        img = cv2.imread(path + "\\" + str(i) + "\\" + j)
        img = cv2.resize(img, (32,32))
        images.append(img)
        class_no.append(i)
print(len(images))
print(len(class_no))

images = np.array(images)
class_no = np.array(class_no)

print(images.shape)
print(class_no.shape)

# veriyi ayırma
x_train, x_test, y_train, y_test = train_test_split(images, class_no, test_size=0.33, random_state=42)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print(images.shape)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

# görselleştirme
fig, axes = plt.subplots(3,1,figsize=(7,7))
fig.subplots_adjust(hspace=0.5)
sns.countplot(y_train, ax=axes[0])
axes[0].set_title("y_train")

sns.countplot(y_test, ax=axes[1])
axes[1].set_title("y_test")

sns.countplot(y_validation, ax=axes[2])
axes[2].set_title("y_validation")

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return(img)

x_train = np.array(list(map(preProcess, x_train)))
x_test = np.array(list(map(preProcess, x_test)))
x_validation = np.array(list(map(preProcess, x_validation)))

x_train = x_train.reshape(-1,32,32,1)
x_test = x_test.reshape(-1,32,32,1)
x_validation = x_validation.reshape(-1,32,32,1)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, rotation_range=10)
dataGen.fit(x_train)

y_train = to_categorical(y_train, no_classes) # one hot encoder ile aynı sonucu çıkartır
y_test = to_categorical(y_test, no_classes)
y_validation = to_categorical(y_validation, no_classes)

# model
model = Sequential()
model.add(Conv2D(filters= 8, kernel_size=(5,5), activation="relu", input_shape = (32, 32, 1), padding="same")) # 8 = filtre sayısı.     
model.add(MaxPooling2D(pool_size=(2,2))) # piksel ekleme
model.add(Conv2D(filters=16, kernel_size=(3,3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Dropout(0.2)) # seyreltme
model.add(Flatten()) # düzleştirme
model.add(Dense(units=256, activation="relu")) # sınıflandırma işlemi
model.add(Dropout(0.2))
model.add(Dense(units=no_classes, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
hist = model.fit_generator(dataGen.flow(x_train, y_train, batch_size=250), validation_data=(x_validation, y_validation), epochs=15,
                           steps_per_epoch=x_train.shape[0] // 250, shuffle=True)

# kaydet
pick_out = open("Deep Learning\Codes\9. Rakam Tespiti\model_trained_new.p", "wb")
pickle.dump(model, pick_out)
pick_out.close()

# sonuç
plt.plot(hist.history["loss"], label = "Eğitim Loss")
plt.plot(hist.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()

plt.plot(hist.history["accuracy"], label = "Eğitim Accuracy")
plt.plot(hist.history["val_accuracy"], label = "Validation Accuracy")
plt.legend()
plt.show()

score = model.evaluate(x_test, y_test, verbose="1")
print(f"Test Loss : {score[0]}")
print(f"Test Accuracy : {score[1]}")

# görselleştirme
y_pred = model.predict(x_validation)
y_pred_class = np.argmax(y_pred, axis=1)
Y_true = np.argmax(y_validation, axis=1)
cm = confusion_matrix(Y_true, y_pred_class)

f, ax = plt.subplots(figsize=(8,8))
sns.heatmap(cm, annot=True, cmap="Greens", linecolor="gray", fmt=".2f", ax=ax)
plt.xlabel("Predict")
plt.ylabel("True")
plt.show()
