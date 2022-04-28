
import pandas as pd
import statsmodels.api as sm
from pmdarima import auto_arima
# from itsdangerous import json
import numpy as np
import json,os
import torch
from torch import nn
from torch.utils import data

from textblob import TextBlob
from flask import Flask,request,jsonify
from lstm_processor import *
from polarity import *
# from lstm_processor import *
from ARIMA_processor import *
# from lstm_torch_processor import *






app = Flask(__name__)

root_dir='./data/'



from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)

def load_stockdata(company):
  path = root_dir
  stock_filename=path+company+'.csv'
  stock_df=pd.read_csv(stock_filename)
  stock_df['Date']=pd.to_datetime(stock_df['Date'])
  stock_df=stock_df[stock_df['Date']>='2019-01-01']
  stock_df=stock_df[stock_df['Date']<='2021-12-31']

  return stock_df



@app.route("/stockdata/<company>", methods=["GET"])
def data(company):

    stock_df = load_stockdata(company)
    stock_df=stock_df[["Date","Close"]]
    stock_df['Date'] = pd.to_datetime(stock_df['Date'],unit='s')
    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date
    stock_json=stock_df.to_json(orient='records')
    return stock_json

@app.route("/ARIMA/1/<company>", methods=["GET"])
def arima(company):
    stock_df = load_stock_data(company)
    stock_price = extract_hist_price(stock_df)


    price_log = np.log(stock_price)  
    price_log_shift = price_log - price_log.shift()
    price_log_shift.dropna(inplace=True)
    print(get_stationarity(price_log_shift))

    split_date = '2021-7-31'

    train_df,test_df = train_test_split(price_log_shift)
    stepwise_fit = auto_arima(price_log_shift['Close'], trace=True,suppress_warnings=True)

    model=sm.tsa.ARIMA(price_log_shift['Close'],order=(2,0,0))
    model=model.fit()
    start=len(train_df)
    end=len(train_df)+len(test_df)-1
    pred = model.predict(start=start,end=end).rename('1-day ARIMA Predictions')

    shift = price_log.shift()
    shift_log_test = shift.loc[shift.index > split_date]

    ad = np.array(shift_log_test.Close)
    pred_arr = np.array(pred)
    sum_two = (pred_arr + ad) if ad != 'nan' else pred_arr
    price_pred = np.exp(sum_two)
    true_test = stock_price[stock_price.index > split_date].Close
    test_df["Predict"]=price_pred
    test_df.drop("Close",axis=1,inplace=True)
    states_buy, states_sell, total_gains, invest =buy_stock(test_df.Predict, initial_state = 1,  delay = 4, initial_money = 10000)
    test_df.reset_index(inplace=True)
    test_df['Buy'] = np.nan
    for i in states_buy:
      test_df['Buy'].iloc[i] = test_df['Predict'].iloc[i]
    test_df['Sell']=np.nan
    for j in states_sell:
      test_df['Sell'].iloc[j]=test_df['Predict'].iloc[j]    
    merged_df=pd.merge(stock_price,test_df, how='outer', on="Date")
    #merged_df=pd.to_datetime(merged_df["Date"], unit='s')
    resp_json=merged_df.reset_index().to_json(orient='records',index=True)

    return resp_json



@app.route("/ARIMA/2/<company>", methods=["GET"])
def arima2(company):
    stock_df = load_stock_data(company)
    stock_price = extract_hist_price(stock_df)


    price_log = np.log(stock_price)  
    price_log_shift = price_log - price_log.shift()
    price_log_shift.dropna(inplace=True)
    print(get_stationarity(price_log_shift))

    split_date = '2021-7-31'

    # train_df,test_df = train_test_split(price_log_exp_decay)
    train_df,test_df = train_test_split(price_log_shift)
    stepwise_fit = auto_arima(price_log_shift['Close'], trace=True,suppress_warnings=True)
    # print(stepwise_fit)

    # model=ARIMA(price_log_exp_decay['Close'],order=(1,0,0))
    model=sm.tsa.ARIMA(price_log_shift['Close'],order=(1,0,0))
    model=model.fit()
    model.summary()

    train_ar = train_df['Close'].values
    test_ar = test_df['Close'].values

    # https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
    history = [x for x in train_ar]
    print(type(history))
    predictions = list()
    for t in range(0,len(test_ar),2):
        model = sm.tsa.ARIMA(history, order=(2,0,1))
        model_fit = model.fit()
        output = model_fit.forecast(steps=2)
        output=list(output)

        if(t<len(test_ar)):
          yhat1,yhat2 = output[0],output[1]
          predictions.append(yhat1)
          
          obs1= test_ar[t]
          history.append(obs1)
          if(t+1<len(test_ar)):
            obs2=test_ar[t+1]
            history.append(obs2)
            predictions.append(yhat2)
          #print('predicted=%f, expected=%f' % (yhat1, obs1))
    shift = price_log.shift()
    shift_log_test = shift.loc[shift.index > split_date]

    ad = np.array(shift_log_test.Close)
    pred_arr = np.array(predictions)
    sum_two = (pred_arr + ad) if ad != 'nan' else pred_arr
    price_pred = np.exp(sum_two)

    # true stock price
    true_test = stock_price[stock_price.index > split_date].Close
    test_df["Predict"]=price_pred
    test_df.drop("Close",axis=1,inplace=True)
    states_buy, states_sell, total_gains, invest =buy_stock(test_df.Predict, initial_state = 1,  delay = 4, initial_money = 10000)
    test_df.reset_index(inplace=True)
    test_df['Buy'] = np.nan
    for i in states_buy:
      test_df['Buy'].iloc[i] = test_df['Predict'].iloc[i]
    test_df['Sell']=np.nan
    for j in states_sell:
      test_df['Sell'].iloc[j]=test_df['Predict'].iloc[j]    
    merged_df=pd.merge(stock_price,test_df, how='outer', on="Date")

    #merged_df=pd.to_datetime(merged_df["Date"], unit='s')
    resp_json=merged_df.reset_index().to_json(orient='records',index=True)

    return resp_json



@app.route("/ARIMA/3/<company>",methods=["GET"])
def arima3(company):
    stock_df = load_stock_data(company)
    stock_price = extract_hist_price(stock_df)


    price_log = np.log(stock_price)  
    price_log_shift = price_log - price_log.shift()
    price_log_shift.dropna(inplace=True)
    print(get_stationarity(price_log_shift))

    split_date = '2021-7-31'

    # train_df,test_df = train_test_split(price_log_exp_decay)
    train_df,test_df = train_test_split(price_log_shift)
    stepwise_fit = auto_arima(price_log_shift['Close'], trace=True,suppress_warnings=True)
    # print(stepwise_fit)

    # model=ARIMA(price_log_exp_decay['Close'],order=(1,0,0))
    model=sm.tsa.ARIMA(price_log_shift['Close'],order=(3,0,0))
    model=model.fit()
    model.summary()

    train_ar = train_df['Close'].values
    test_ar = test_df['Close'].values

    # https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
    history = [x for x in train_ar]
    print(type(history))
    predictions = list()
    for t in range(0,len(test_ar),3):
        model = sm.tsa.ARIMA(history, order=(3,0,1))
        model_fit = model.fit()
        output = model_fit.forecast(steps=3)
        if(t<len(test_ar)):
          yhat1,yhat2,yhat3 = output[0],output[1],output[2]
          predictions.append(yhat1)
          
          obs1= test_ar[t]
          history.append(obs1)
          if(t+1<len(test_ar)):
            obs2=test_ar[t+1]
            history.append(obs2)
            predictions.append(yhat2)
            if(t+2<len(test_ar)):
              obs3=test_ar[t+2]
              history.append(obs3)
              predictions.append(yhat3)

          #print('predicted=%f, expected=%f' % (yhat1, obs1))

    shift = price_log.shift()
    shift_log_test = shift.loc[shift.index > split_date]

    ad = np.array(shift_log_test.Close)
    pred_arr = np.array(predictions)
    sum_two = (pred_arr + ad) if ad != 'nan' else pred_arr
    price_pred = np.exp(sum_two)

    # true stock price
    true_test = stock_price[stock_price.index > split_date].Close
    test_df["Predict"]=price_pred
    test_df.drop("Close",axis=1,inplace=True)
    states_buy, states_sell, total_gains, invest =buy_stock(test_df.Predict, initial_state = 1,  delay = 4, initial_money = 10000)
    test_df.reset_index(inplace=True)
    test_df['Buy'] = np.nan
    for i in states_buy:
      test_df['Buy'].iloc[i] = test_df['Predict'].iloc[i]
    test_df['Sell']=np.nan
    for j in states_sell:
      test_df['Sell'].iloc[j]=test_df['Predict'].iloc[j]    
    
    merged_df=pd.merge(stock_price,test_df, how='outer', on="Date")
    #merged_df=pd.to_datetime(merged_df["Date"], unit='s')
    resp_json=merged_df.reset_index().to_json(orient='records',index=True)

    return resp_json





@app.route("/LSTM/<target_size>/<company>", methods=["GET"])
def predict(target_size,company):

  val=get_lstm_results(company,int(target_size))
  return val

@app.route("/tweetper/<company>", methods=["GET"])
def tweets_per(company):
  tweets_dict=get_tweets_percentages(company)
  return jsonify(tweets_dict)

# AAPL_MODEL=torch.load("./save_and_load/lstm_model_AAPL/LSTM_AAPL_5.pth")


@app.route("/LSTM/<company>/<target>",methods=['GET','POST'])
def LSTM(company,target):
  output_json_path="./"+company+"out.json"
  os.system("python lstm_torch_processor.py {} {}".format(company,output_json_path))
  with open(output_json_path) as fp:
      k=json.load(fp)
  os.remove(output_json_path)
  res=[]
  for i in k[company]:
    res.append(i[int(target)])

  return jsonify(res)




# start the flask app, allow remote connections 
if __name__=="__main__":

  app.run(host='0.0.0.0',debug=True)