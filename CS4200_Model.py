#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:20:13 2020

@author: mike
"""

import tensorflow as tf, pandas as pd, numpy as np, matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense, Dropout
import time, shutil
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import pandas_datareader.data as pdr

#Quick Parameters:
window_size = 10
test_size = 5
val_size = 20
batchsize = 10
start_date = datetime(2017,1,1)
end_date = datetime(2020,11,30)
epoch=20
log_dir= "/home/mike/Desktop/LSTM"

#%% Import data and add necessary columns

def fetch_data(ticker, start, end):
    #Fetches the pricing data of ticker from yahoo 
    try:
        df=pdr.DataReader(ticker, 'yahoo', start, end)
        df.to_csv(f'stock_details/{ticker}.csv')
    except:
        print("Error")
    return df
  
rawdata=fetch_data('AMZN',start_date, end_date)

#%% Prepare data for split/training

#Feature scaling, one scaler per column
column_scalers={}
data=pd.DataFrame(index=rawdata.index)

for column in rawdata.columns.values:
    scaler = MinMaxScaler(feature_range=(0,1))
    data.loc[:,column] = scaler.fit_transform(np.expand_dims(rawdata[column].values, axis=1))
    column_scalers[column] = scaler 

x_samples, y_samples = [],[]
 
for i in range(window_size , len(data) , 1):
    # Using a 'sliding window' model, partition data into arrays of size 'window_size'
    x_sample = np.array(data[i-window_size:i])
    y_sample = np.array(data[i:i+1]['Adj Close'])
    x_samples.append(x_sample)
    y_samples.append(y_sample)
 
# Convert lists to arrays
x_data, y_data = np.array(x_samples), np.array(y_samples)

tomorrow = np.array(data[len(data)-window_size:])                               
tomorrow = tomorrow.reshape(1,len(tomorrow),len(tomorrow[0]))
#%% Split data into training, testing, and validation sets

train_size = x_data.shape[0] - test_size - val_size

x_train = x_data[0:train_size]
y_train = y_data[0:train_size]

x_val = x_data[train_size:train_size+val_size]
y_val = y_data[train_size:train_size+val_size]

train_size += val_size

x_test = x_data[train_size:train_size+test_size]
y_test = y_data[train_size:train_size+test_size]

#%% Build the Model

#Define input shape for LSTM
features = x_data.shape[2]

#Initialize the model
model = tf.keras.Sequential()
model.add(LSTM(units = 200,activation='relu', input_shape = (window_size, features)))
model.add(Dense(units=200))
model.add(Dense(units=100))
model.add(Dense(units=1))

model.compile(optimizer = 'adam', loss = 'mse', metrics='mean_absolute_error')
  
#%% Train the model

#Create tensorboard log files
logdir=log_dir + datetime.now().strftime("%Y%m%d")
try: shutil.rmtree(logdir)
except: pass
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Measuring the time taken by the model to train
start=time.time()
 
#Fitting the model to the Training set using GPU0
with tf.device('/GPU:0'):
    history = model.fit(x_train,
                        y_train,
                        batch_size=batchsize,
                        epochs=epoch,
                        validation_data=(x_val,y_val),
                        callbacks=[tensorboard_callback])
 
end=time.time()
print('**** Training Duration: ', round((end-start)/60), 'Minutes ****')

#%% Assess Accuracy

# Making predictions on test data
pred_index = np.where(data.columns=='Adj Close')[0][0]

predicted_Price = model.predict(x_test)
predicted_Price = column_scalers['Adj Close'].inverse_transform(predicted_Price)
 
# Getting the original price values for testing data
orig=y_test
orig=column_scalers['Adj Close'].inverse_transform(y_test)
 
plt.plot(predicted_Price, color = 'blue', label = 'Predicted Price')
plt.plot(orig, color = 'lightblue', label = 'Original Price')
 
plt.title('Stock Price Predictions')
plt.xlabel('Trading Date')
plt.xticks(range(x_test.shape[0]), data[len(data)-x_test.shape[0]:len(data)].index)
plt.ylabel('Stock Price')
 
plt.legend()
fig=plt.gcf()
fig.set_figwidth(20)
fig.set_figheight(6)
plt.show()

#%% Show model predictions on the full dataset

# Generating predictions on full data
x_predictions=column_scalers['Adj Close'].inverse_transform(model.predict(x_data))
#TestPredictions=column_scalers['Adj Close'].inverse_transform(model.predict(x_test))
 
#FullDataPredictions=np.append(TrainPredictions, TestPredictions)
y_actual=column_scalers['Adj Close'].inverse_transform(y_data)
 
# plotting the full data
plt.plot(x_predictions, color = 'blue', label = 'Predicted Price')
plt.plot(y_actual , color = 'lightblue', label = 'Original Price')
 
plt.title('Stock Price Predictions')
plt.xlabel('Trading Date')
plt.ylabel('Stock Price')
plt.legend()
fig=plt.gcf()
fig.set_figwidth(20)
fig.set_figheight(8)
plt.show()

#%% Stats

#Prepare gradient dataframe from historic data
df1=pd.DataFrame(rawdata['Adj Close'][window_size:])
df1.index=rawdata.index[window_size:]
df1['Gradient']=df1['Adj Close']-df1['Adj Close'].shift(1)
df1=df1['Gradient'].transform(lambda x: 1 if x > 0 else 0)

#Prepare gradient dataframe from predictions
temp=column_scalers['Adj Close'].inverse_transform(model.predict(x_data))
test_pred=pd.DataFrame(temp)
test_pred.index=rawdata.index.values[window_size:]
test_pred['Gradient']=test_pred[0]-test_pred[0].shift(1)
test_pred=test_pred['Gradient'].transform(lambda x: 1 if x > 0 else 0)

grad_acc = sum(df1 == test_pred)/len(df1)
grad_acc_test = sum(df1[len(test_pred)-test_size:] == test_pred[len(test_pred)-test_size:])/test_size
tomorrow_price = column_scalers['Adj Close'].inverse_transform(model.predict(tomorrow))[0][0]

print('Gradient Accuracy (all data): ', round(grad_acc,3))
print('Gradient Accuracy (test data): ', round(grad_acc_test,3))
print('Tomorrows Price: ', round(tomorrow_price,2))

print(rawdata['Adj Close'].head(10), x_predictions[0:10])

