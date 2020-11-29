#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:20:13 2020

@author: mike
"""

import tensorflow as tf, pandas as pd, numpy as np, matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Dense, Dropout
import json, sklearn, time, tensorflow_datasets
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from datetime import datetime

#%% Import and prepare data

#Quick Parameters:
window_size = 5
test_size = 5
batchsize = 9
date = datetime(2020,1,1)

#Load the data from .csv
rawdata = pd.read_csv('AMZN.csv')

#Set index as 'Date' and convert to datetime, allows or limiting of date range.
rawdata=rawdata.set_index('Date')
rawdata.index=pd.to_datetime(rawdata.index)
rawdata=rawdata.loc[rawdata.index > date]

#Feature Scaling
column_scalers={}
data=pd.DataFrame(index=rawdata.index)

for column in rawdata.columns.values:
    scaler = MinMaxScaler(feature_range=(0,1))
    data.loc[:,column] = scaler.fit_transform(np.expand_dims(rawdata[column].values, axis=1))
    column_scalers[column] = scaler 

x_samples = []
y_samples = []
 
# Using a 'sliding window' model, partition data into arrays of size 'window_size'
for i in range(window_size , len(data) , 1):
    x_sample = np.array(data[i-window_size:i])
    y_sample = np.array(data[i:i+1]['Adj Close'])
    x_samples.append(x_sample)
    y_samples.append(y_sample)
 
# Reshape the Input as a 3D (number of samples, Time Steps, Features)
x_data = np.array(x_samples)                                  
x_data = x_data.reshape(x_data.shape[0],x_data.shape[1], x_data.shape[2])

y_data=np.array(y_samples)
y_data=y_data.reshape(y_data.shape[0], y_data.shape[1])

#%% Split data into training and testing

train_size = x_data.shape[0] - 2*test_size

x_train = x_data[0:train_size]
y_train = y_data[0:train_size]

x_val = x_data[train_size:train_size+test_size]
y_val = y_data[train_size:train_size+test_size]

train_size += test_size

x_test = x_data[train_size:train_size+test_size]
y_test = y_data[train_size:train_size+test_size]

#%% Build the Model

# Defining Input shapes for LSTM
TimeSteps=window_size
TotalFeatures=x_data.shape[2]

#Initialize the model
model = tf.keras.Sequential()

model.add(LSTM(units = 64, activation='relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 64, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 64, activation='relu'))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'huber_loss', metrics='mean_absolute_error')
  
#%% Train the model

#Create tensorboard log files
logdir="/home/mike/Desktop/LSTM" + datetime.now().strftime("%Y%m%d")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Measuring the time taken by the model to train
StartTime=time.time()
 
# Fitting the RNN to the Training set using GPU0
with tf.device('/GPU:0'):
    history = model.fit(x_train,
                        y_train,
                        batch_size=batchsize,
                        epochs=20,
                        validation_data=(x_val,y_val),
                        callbacks=[tensorboard_callback])
 
EndTime=time.time()
print("## Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ##')


#%% Assess Accuracy
# Making predictions on test data
pred_index = np.where(data.columns=='Adj Close')[0][0]

predicted_Price = model.predict(x_test)
predicted_Price = column_scalers['Adj Close'].inverse_transform(predicted_Price)
 
# Getting the original price values for testing data
orig=y_test
orig=column_scalers['Adj Close'].inverse_transform(y_test)
 
# Accuracy of the predictions
print('Accuracy:', 100 - (100*(abs(orig-predicted_Price)/orig)).mean())


plt.plot(predicted_Price, color = 'blue', label = 'Predicted Price')
plt.plot(orig, color = 'lightblue', label = 'Original Price')
 
plt.title('Stock Price Predictions')
plt.xlabel('Trading Date')
plt.xticks(range(x_test.shape[0]), data.tail(x_test.shape[0]).index)
plt.ylabel('Stock Price')
 
plt.legend()
fig=plt.gcf()
fig.set_figwidth(20)
fig.set_figheight(6)
plt.show()

#%% 
# Generating predictions on full data
TrainPredictions=column_scalers['Adj Close'].inverse_transform(model.predict(x_train))
TestPredictions=column_scalers['Adj Close'].inverse_transform(model.predict(x_test))
 
FullDataPredictions=np.append(TrainPredictions, TestPredictions)
FullDataOrig=column_scalers['Adj Close'].inverse_transform(y_samples)
 
# plotting the full data
plt.plot(FullDataPredictions, color = 'blue', label = 'Predicted Price')
plt.plot(FullDataOrig , color = 'lightblue', label = 'Original Price')
 
 
plt.title('Stock Price Predictions')
plt.xlabel('Trading Date')
plt.ylabel('Stock Price')
plt.legend()
fig=plt.gcf()
fig.set_figwidth(20)
fig.set_figheight(8)
plt.show()

