# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 16:55:39 2018

@author: japes
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('Google_Stock_Price_Train.csv')

training_set= dataset.iloc[:,1:2]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
scaled_training_set = sc.fit_transform(training_set)

# trining a datastructure with  timesteps and 1 output

X_train=[]
y_train=[]
for i in range(60,1258):
    X_train.append(scaled_training_set[i-60:i,0])
    y_train.append(scaled_training_set[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)

#reshaping

X_train = np.reshape(X_train, newshape=(1198,60, 1))  


#building rnn
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

regressor= Sequential()
regressor.add(LSTM(units=50, return_sequences=True , input_shape=(60,1)))
regressor.add(Dropout(rate=.2))

regressor.add(LSTM(units=50, return_sequences=True ))
regressor.add(Dropout(rate=.2))

regressor.add(LSTM(units=50, return_sequences=True ))
regressor.add(Dropout(rate=.2))

regressor.add(LSTM(units=50, return_sequences=False ))
regressor.add(Dropout(rate=.2))

#output layer
regressor.add(Dense(units=1))

#compiling the rnn
regressor.compile(optimizer='adam', loss='mean_squared_error')

#fitting the rnn to training set

regressor.fit(X_train,y_train,epochs=100,batch_size=64)


dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=dataset_test.iloc[:,1:2].values


dataset_total= pd.concat((dataset['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs=sc.transform(inputs)



X_test=[]
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)
X_test = np.reshape(X_test, newshape=(20,60, 1))

#predict stock price for 2017
predicted_google_stock_price= regressor.predict(X_test)
predicted_google_stock_price=sc.inverse_transform(predicted_google_stock_price)

#visualising the results
plt.plot(real_stock_price,color='blue', label='real stock price for 2017')
plt.plot(predicted_google_stock_price,color='red',label='predicted stock price')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()

regressor.save('google_stock.h5')
regressor.optimizer