#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 17:10:15 2020

@author: hexaguy
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataframe = pd.read_csv("iris.csv")

dataset = dataframe.values

X = dataset[: , 0:4].astype(float)

y = dataset[: , 4]

from sklearn.preprocessing import LabelEncoder,StandardScaler
from keras.utils import np_utils


encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

dummy_y = np_utils.to_categorical(encoded_y)

sc = StandardScaler()
X = sc.fit_transform(X)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


classifier = Sequential()
classifier.add(Dense(units=8, input_shape=(4,),activation='relu'))
classifier.add(Dense(units=3,activation='softmax'))
classifier.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
classifier.fit(X, dummy_y, batch_size=5, epochs=200)


y_pred = classifier.predict(X)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(dummy_y.argmax(axis=1),np.round(y_pred).argmax(axis=1))

