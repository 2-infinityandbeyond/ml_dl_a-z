import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])

labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.fit_transform(X_test)

#make ANN

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier =Sequential()

#adding input layer and first hidden layer
classifier.add(Dense(units=6,init='uniform',activation='relu',input_dim=11))
 
#adding second hidden layer
classifier.add(Dense(units=6,init='uniform',activation='relu'))

# adding the output layer
classifier.add(Dense(units=1,init='uniform',activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the ann to training set
classifier.fit(X_train,y_train,batch_size=20
               ,epochs=20)

#predicting the test set
y_pred=classifier.predict(X_test)
y_pred=(y_pred>.57)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


acc_racy=(cm[1,1]+cm[0,0])/2000




