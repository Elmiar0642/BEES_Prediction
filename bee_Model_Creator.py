import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import os,pprint
import random
import gc
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='__' #YOUR GPU ID
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split as TTS
from keras import layers,models,optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array,load_img
import tensorflow as tf
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer

EPOCHS = 100
INIT_LR = 1e-3
BS = 1000
default_image_size = tuple((150,150))
width=150
height=150
depth=3
inputShape=(150,150,3)

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))#,dtype=int))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))


#flatten
model.add(layers.Flatten())


#hidden layers
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.5))

#output layers
model.add(layers.Dense(4,activation='softmax'))#'sigmoid'


model.summary()

model.compile(loss='categorical_crossentropy',#'hinge'
              optimizer='adam',#optimizers.SGD(lr=1e-4),
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,)
val_datagen=ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('',#YOUR PATH : 'A:\\Arima\\PROJECTS\\Outbox\\PRJ\\HoneyBee\\bees\\train',
                                                    target_size=(150,150),
                                                    batch_size=64,
                                                    class_mode='categorical')


val_generator = val_datagen.flow_from_directory('',#YOUR PATH : 'A:\\Arima\\PROJECTS\\Outbox\\PRJ\\HoneyBee\\bees\\validate',
                                                    target_size=(150,150),
                                                    batch_size=64,
                                                    class_mode='categorical')

history = model.fit(train_generator,
                              validation_data = val_generator,
                              validation_steps=len(val_generator),
                              steps_per_epoch=len(train_generator),
                              epochs=100,verbose=1)


save_dir = '' #YOUR PATH : "A:\\Arima\\PROJECTS\\CNN\\VGG-16\\bees_model.h5"
model.save(save_dir)

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

#Train and test accuracy
plt.plot(epochs,acc,'b',label='Training Accuracy',color="red")
plt.plot(epochs,val_acc,'b',label='Testing Accuracy')
plt.title("Train and test accuracy")
plt.legend()

plt.figure()

#Train and test loss
plt.plot(epochs,loss,'b',label='Training loss',color="red")
plt.plot(epochs,val_loss,'b',label='Testing loss')
plt.title("Train and test loss")
plt.legend()

plt.show()

scores = model.evaluate(x_train, y_train)
print("Accuracy is :{}\n\nFinished Ready to Deploy, MASTER!".format(str(scores[1]*100)))
