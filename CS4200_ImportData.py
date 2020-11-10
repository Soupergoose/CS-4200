#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import bs4 as bs
import datetime as dt
import math, sklearn, pickle, requests, os
import pandas_datareader.data as pdr

#These functions are based on the code used by Abhijit Roy at the following link:
#https://towardsdatascience.com/stock-price-prediction-based-on-deep-learning-3842ef697da0

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
  with open("tickers.pickle",'wb') as writer:
      #writes the contents of 'ticker' to a .pickle file
      pickle.dump(tickers, writer)
  return tickers



def fetch_data():
  #Fetches the pricing data of each ticker from yahoo

  with open("tickers.pickle",'rb') as f:
    tickers=pickle.load(f)
    
  if not os.path.exists('stock_details'):
    os.makedirs('stock_details')
  
  start= dt.datetime(2010,1,1)
  end=dt.datetime(2020,11,6)

  count=0
  for ticker in tickers:

    if count==200:
      #limits the number of files produced
      break
    count+=1
    print(ticker)
  
    try:
      #Using pandas_datareader, download the ticker data between 'start' and 'end' from yahoo finance.
      df=pdr.DataReader(ticker, 'yahoo', start, end)
      df.to_csv('stock_details/{}.csv'.format(ticker))
    except:
      print("Error")
    continue


save_tickers()
fetch_data()