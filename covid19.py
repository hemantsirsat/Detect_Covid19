import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

data_dir = "/home/hemant/Desktop/covid-chestxray-dataset/dataset/"
categories = ["covid19","normal"]

for category in categories:
    path = os.path.join(data_dir,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        #plt.imshow(img_array,cmap="gray")
        #plt.show()
        break
    break

#print(img_array.shape)
IMG_SIZE = 350
new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
#plt.imshow(new_array,cmap="gray")
#plt.show()

training_data = []


#0 = covid, 1 = normal
def create_training_data():
    for category in categories:
        path = os.path.join(data_dir,category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            training_data.append([new_array,class_num])
            #plt.imshow(img_array,cmap="gray")
            #plt.show()

create_training_data()

print(len(training_data))

import random
random.shuffle(training_data)
'''for sample in training_data[:10]:
    print(sample[1])
'''
X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)
    
X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)

import pickle
pickle_out=open("x.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open("y.pickle","wb")
pickle.dump(Y,pickle_out)
pickle_out.close()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Flatten

X = pickle.load(open("x.pickle","rb"))
Y = pickle.load(open("y.pickle","rb"))

X = X/255.0

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=['accuracy'])

model.fit(X,Y,batch_size=100,epochs = 10,validation_split=0.1)