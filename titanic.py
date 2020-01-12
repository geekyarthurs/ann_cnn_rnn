

# %% IMPORTING MAIN LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



training_df = pd.read_csv("train.csv")



training_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)


X = training_df.values[:, 1:]
y = training_df["Survived"].values


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values= np.nan, strategy='mean')
m_imp = SimpleImputer(missing_values= np.nan , strategy='most_frequent')

ages = X[: , 2].reshape((-1,1))
ages = imp.fit_transform(ages)
ages = ages.reshape((1,-1)).round(decimals=2)
X[: , 2] = ages


ages = X[: , 6].reshape((-1,1))
ages = m_imp.fit_transform(ages)
ages = ages.reshape((1,-1))

X[: , 6] = ages
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

encoder = LabelEncoder()

X[: , 1] = encoder.fit_transform(X[ : ,1])

transformer = ColumnTransformer(transformers = [("Onehot", OneHotEncoder(), [6] )], remainder='passthrough')


X = transformer.fit_transform(X.tolist())
X = X.astype('float64')

# Importing the Keras libraries and packages
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

classifier = Sequential()
classifier.add(Dense(units=5, input_shape=(9,), activation='relu'))
classifier.add(Dense(units=8, activation='relu'))

classifier.add(Dense(units=1,  activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X, y, batch_size=16, epochs=100)


# %% TESTING


testing_df = pd.read_csv('test.csv')



testing_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)


test_x = testing_df.values

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values= np.nan, strategy='mean')
m_imp = SimpleImputer(missing_values= np.nan , strategy='most_frequent')

ages = test_x[: , 2].reshape((-1,1))
ages = imp.fit_transform(ages)
ages = ages.reshape((1,-1)).round(decimals=2)
test_x[: , 2] = ages

ages = test_x[: , 6].reshape((-1,1))
ages = m_imp.fit_transform(ages)
ages = ages.reshape((1,-1))

test_x[: , 6] = ages

test_x[: , 1] = encoder.fit_transform(test_x[ : ,1])

transformer = ColumnTransformer(transformers = [("Onehot", OneHotEncoder(), [6] )], remainder='passthrough')


test_x = transformer.fit_transform(test_x.tolist())
test_x = test_x.astype('float64')


pred_y = classifier.predict(test_x)


pred_y = pred_y.round(decimals=0).astype(int)
temp_df = pd.read_csv('test.csv')["PassengerId"].values.reshape((-1,1))
final_array = np.concatenate((temp_df, pred_y), axis=1)
final_df = pd.DataFrame(data=final_array)
final_df.to_csv("final.csv", index=False)
