"""
Bu projede yapay zekada transfer learning işlemi uygulanmıştır. Proje içinde yer alan yerleri değiştirmeniz gerekmektedir.
"""
from keras import layers
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator


model = VGG16(include_top=False, input_shape=(224,224,3))

for layer in model.layers:
    layer.trainable = False

flattened_layer = layers.Flatten()(model.output) # flatten katmanında vgg modelini kullandık
full_connect_layer = layers.Dense(512, activation="relu")(flattened_layer) # full connected katmanına dahil ettik
output_layer = layers.Dense(1, activation="sigmoid")(full_connect_layer) # çıktı katmanına dahil ettik

model = Model(inputs=model.inputs, outputs=output_layer)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]) 
print(model.summary()) # modelin çıktısını inceleyelim

train_directory = "train resimlerinin olduğu yol"
validation_directory = "validation resimlerinin olduğu yol"

# Data Augmentation
train_dataGen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True, rotation_range=45, height_shift_range=0.2, width_shift_range=0.2, fill_mode="nearest")
validation_dataGen = ImageDataGenerator(rescale=1./255)

train_generator = train_dataGen.flow_from_directory(train_directory, target_size=(224,224), batch_size=64, class_mode="binary")
validation_generator = validation_dataGen.flow_from_directory(validation_directory, target_size=(224,224), batch_size=16, class_mode="binary")

# Eğitim
hist = model.fit(train_generator, batch_size=250, validation_data=validation_generator, epochs=50, verbose=2)

# Modeli haydet
model.save("model.h5")