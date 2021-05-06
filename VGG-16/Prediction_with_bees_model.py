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
os.environ['CUDA_VISIBLE_DEVICES']='__' # set your own GPU number
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
from sklearn.metrics import confusion_matrix,classification_report

model_path = "A:\\Arima\\PROJECTS\\CNN\\VGG-16\\bees_model.h5" #input("Model Path")

model = load_model(model_path)

model.summary()

test_datagen=ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
  "",
#YOUR PATH IN THIS FORMAT :'A:\\Arima\\PROJECTS\\Outbox\\PRJ\\HoneyBee\\bees\\test',
target_size=(150, 150),
batch_size=20,
class_mode='categorical',
shuffle = False)
test_loss, test_acc = model.evaluate(test_generator, steps=50)

print('test acc:', test_acc)

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
