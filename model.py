#importing libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import tensorflow_hub as hub
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau

train_df = pd.read_csv('/data/sign_mnist_train.csv') #read train dataset
test_df = pd.read_csv('/data/sign_mnist_test.csv') #read test dataset
y = test_df['label'] #make a copy of the test dataset labels for model evaluation
y_train = train_df['label'] #store train dataset labels in a seperate list
y_test = test_df['label'] #store train dataset labels in a seperate list
del train_df['label'] #delete labels column from train dataset
del test_df['label'] #delete labels column from test dataset

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train) #binarize train labels
y_test = label_binarizer.fit_transform(y_test) #binarize test labels

x_train = train_df.values #input matrix from training dataset
x_test = test_df.values #input matrix from test dataset

x_train = x_train/255 #perform grayscale normailization on training dataset
x_test = x_test/255 #perform grayscale normailization on test dataset

x_train = x_train.reshape(-1,28,28,1) #reshape training data for CNN input
x_test = x_test.reshape(-1,28,28,1) #reshape test data for CNN input

#perform data augmentation
datagen = ImageDataGenerator(
        rotation_range=10,  #randomly rotate images in the 10degrees range
        zoom_range = 0.1, #randomly zoom images by 10%
        width_shift_range=0.1,  #randomly shift images horizontally by 10%
        height_shift_range=0.1,  #randomly shift images vertically by 10%
        horizontal_flip=False,  #randomly flip images
        vertical_flip=False)  #randomly flip images
datagen.fit(x_train)

#define CNN model
model = Sequential()
model.add(Conv2D(75,(3,3),strides=1,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2),strides=2,padding='same'))
model.add(Conv2D(50,(3,3),strides=1,padding='same',activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2),strides=2,padding='same'))
model.add(Conv2D(25,(3,3),strides=1,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2),strides=2,padding='same'))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(24,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#train CNN model on training data
history = model.fit(datagen.flow(x_train,y_train,batch_size=128),epochs=20,validation_data=(x_test,y_test))

print("Accuracy of the model is: ",model.evaluate(x_test,y_test)[1]*100,"%") #model final accuracy

#saves training and testing accuracy and loss graphs
epochs = [i for i in range(20)]
plt.figure(figsize=(15,15))
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
ax[0].plot(epochs,train_acc,color="#cf5151",label='Training Accuracy')
ax[0].plot(epochs,val_acc,color="#4456b3",label='Testing Accuracy')
ax[0].set_title('Training & Validation Accuracy',fontsize=12)
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")
ax[1].plot(epochs,train_loss,color="#cf5151",label='Training Loss')
ax[1].plot(epochs,val_loss,color="#4456b3",label='Testing Loss')
ax[1].set_title('Training & Validation Loss',fontsize=12)
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.savefig("/deliver/accuracy.jpg")
plt.show()

#precision, recall and f1 score for each class
predictions = model.predict_classes(x_test)
for i in range(len(predictions)):
    if(predictions[i]>=9):
        predictions[i] += 1
predictions[:5]   
classes = ["Class "+str(i) for i in range(25) if i!=9]
print(classification_report(y,predictions,target_names=classes))

#saving tflite model for flutter application
tf.keras.models.save_model(model,"/savedmodels/asl_aplhabet",overwrite=True,include_optimizer=True,save_format='tf',signatures=None,)
asl_model = tf.keras.models.load_model("/savedmodels/asl_aplhabet",custom_objects={'KerasLayer':hub.KerasLayer},compile=True)
# Get the concrete function from the Keras model.
run_model = tf.function(lambda x:asl_model(x))
# Save the concrete function.
concrete_func = run_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape,model.inputs[0].dtype))
# Convert the model
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converted_tflite_model = converter.convert()
open("/savedmodels/asl.tflite","wb").write(converted_tflite_model)
# Convert the model to quantized version with post-training quantization
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
open("/savedmodels/asl_quant.tflite","wb").write(tflite_quant_model)