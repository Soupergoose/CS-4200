#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import bs4 as bs
import datetime as dt
import math, sklearn, json, requests, os
import pandas_datareader.data as pdr

#These functions are based on the code used by Abhijit Roy at the following link:
#https://towardsdatascience.com/stock-price-prediction-based-on-deep-learning-3842ef697da0
test_ticker = 'AMZN'

def save_tickers():
  # Downloads a list of the S&P500 companies from wikipedia
  response=requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
  soup=bs.BeautifulSoup(response.text)
  table=soup.find('table',{'class':'wikitable sortable'})
  tickers=[]
  for record in table.findAll('tr')[1:]:
      # Loops over each record in the table and appends the ticker to 'tickers'
      ticker=record.findAll('td')[0].text[:-1]
      tickers.append(ticker)
  with open("tickers.json",'w', encoding="utf8") as writer:
      #writes the contents of 'ticker' to a .json file
      json.dump(tickers, writer)
  return tickers



def fetch_data():
  #Fetches the pricing data of each ticker from yahoo (12 minutes)
  with open("tickers.json",'rb') as data:
    tickers=json.load(data)
    
  if not os.path.exists('stock_details'):
    os.makedirs('stock_details')
  
  start= dt.datetime(2010,1,1)
  end=dt.datetime(2020,11,6)

  count=0
  for ticker in tickers:

    if count==500:
      break
    count+=1
    print(ticker)
    try:
      df=pdr.DataReader(ticker, 'yahoo', start, end)
      df.to_csv(f'stock_details/{ticker}.csv')
    except:
      print("Error")
      continue




def Generate_Training_File():
    #Generates training data file from stock data files
    with open("tickers.json",'rb') as data:
        tickers=json.load(data)
   
        main_df=pd.DataFrame()
    
        for count,ticker in enumerate(tickers):
            if test_ticker in ticker:
                continue
            if not os.path.exists(f'stock_details/{ticker}.csv'):
                continue
            df=pd.read_csv(f'stock_details/{ticker}.csv')
            df.set_index('Date',inplace=True)
       
            df.rename(columns={'Adj Close': ticker}, inplace=True)
            df.drop(['Open','High','Low',"Close",'Volume'],axis=1,inplace=True)
    
            if main_df.empty: main_df=df
            
            else: main_df=main_df.join(df,how='outer')
       
        print(main_df.head())
    main_df.to_csv('Dataset_temp.csv')




save_tickers()
fetch_data()
Generate_Training_File()