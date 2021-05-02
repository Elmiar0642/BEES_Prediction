import cv2
import datetime
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
os.environ['CUDA_VISIBLE_DEVICES']='2'
#pprint.pprint(dict(os.environ),width=1)
#print(os.environ['CUDA_VISIBLE_DEVICES'])
#input()
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
from keras.models import load_model

#labels = ["category0", "category1", "category2", "category3"]

#labelencoder = LabelBinarizer()
#label=labelencoder.fit_transform([0,1,2,3])

#os.environ['CUDA_VISIBLE_DEVICES']='2'
model_path = "A:\\Arima\\PROJECTS\\CNN\\VGG-16\\bees_model.h5" #input("Model Path")
model = load_model(model_path)#'C:/Users/arima/OneDrive/Desktop/ARIMA FOLDER MASTER/PROJECTS/kaggle_api/tomato/weights/Final/model_keras_UPDATE____TWO_____05_12_2020_fifty_epochs.h5')
model.summary()

test_datagen=ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
'A:\\Arima\\PROJECTS\\Outbox\\PRJ\\HoneyBee\\bees\\test',
target_size=(150, 150),
batch_size=20,
class_mode='categorical',
shuffle = False)
test_loss, test_acc = model.evaluate(test_generator, steps=50)
print('test acc:', test_acc)

from sklearn.metrics import confusion_matrix,classification_report
batch_size = 20
num_of_test_samples = 70
predictions = model.predict(test_generator,  num_of_test_samples // batch_size+1)

y_pred = np.argmax(predictions, axis=1)

true_classes = test_generator.classes

class_labels = list(test_generator.class_indices.keys())   

print(class_labels)

print(confusion_matrix(test_generator.classes, y_pred))

report = classification_report(true_classes, y_pred, target_names=class_labels)
print(report)


'''
def img_to_np(DIR,flatten=True):
  #canny edge detection by resizing
  cv_img=mpimg.imread(DIR,0)
  cv_img=cv2.resize(cv_img,(150,150))
  img = np.uint8(cv_img)
  #img = np.uint8((0.2126 * img[:,:,0]) + np.uint8(0.7152 * img[:,:,1]) + np.uint8(0.0722 * img[:,:,2]))
  #flatten it
  if(flatten):
    img=img.flatten()
  return img


path="A:\\Arima\\PROJECTS\\Outbox\\PRJ\\HoneyBee\\bees\\test\\{}".format(random.choice(os.listdir("A:\\Arima\\PROJECTS\\Outbox\\PRJ\\HoneyBee\\bees\\test"))) #"C:/Users/arima/OneDrive/Desktop/ARIMA FOLDER MASTER/PROJECTS/kaggle_api/tomato/real"
files=os.listdir(path)
d=random.choice(files)
print("{}\\{}".format(path,d))
arr=img_to_np("{}\\{}".format(path,d),flatten=False)
arr=arr.reshape(1,150,150,3)
print(model.predict(arr)[0])
print(labelencoder.inverse_transform(model.predict(arr))[0])
print(labels[labelencoder.inverse_transform(model.predict(arr))[0]])
'''
