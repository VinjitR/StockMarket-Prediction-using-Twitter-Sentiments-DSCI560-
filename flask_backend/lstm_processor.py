import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import re 
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from keras.models import Model, model_from_json
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore')

root_dir = './data'

def load_stock_data(company):
  path = root_dir
  stock_filename=path+'/' + company+'.csv'
  stock_df=pd.read_csv(stock_filename)
  stock_df['Date']=pd.to_datetime(stock_df['Date'])
  stock_df=stock_df[stock_df['Date']>='2019-01-01']
  stock_df=stock_df[stock_df['Date']<='2021-12-31']

  return stock_df


def load_tweets_df(company):
  path = root_dir
  tweets_file_name=path+'/' + company+'_tweets.csv'
  tweets_df=pd.read_csv(tweets_file_name,usecols=range(1,5))
  tweets_df.dropna(inplace=True)
  tweets_df['Datetime']=tweets_df['Datetime'].str.split(' ').str.get(0)
  tweets_df['Datetime']=pd.to_datetime(tweets_df['Datetime'])

  return tweets_df


def cleanText(text):
  text=re.sub(r'@[A-Za-z0-9_]+','',text)
  text=re.sub(r'#','',text)
  text=re.sub(r'RT :','',text)
  text=re.sub(r'https?://[A-Za-z0-9./]+','',text)
  text=re.sub(r'\n',' ',text)
  text=re.sub(r'  ',' ',text)

  return text

def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
  return TextBlob(text).sentiment.polarity

def getClass(polarity):
  if polarity<0:
    return 'negative'
  if polarity==0:
    return 'neutral'
  if polarity>0:
    return 'positive'

def mergeData(stock_df,tweets_df,split_date='2021-07-31'):
  merge_df = stock_df.copy()
  for i in range(0,len(stock_df)):
    cur_time = stock_df.iloc[i,:]['Date']
    idx = tweets_df['Datetime'] == cur_time
    tweets_sub = tweets_df.loc[idx]

    tweets_sub['len_chars'] = tweets_sub['Text'].apply(len)
    merge_df.loc[i,'len_chars'] = tweets_sub['len_chars'].mean()
    merge_df.loc[i,'polarity'] = tweets_sub['Polarity'].mean()
    merge_df.loc[i,'subjectivity'] = tweets_sub['Subjectivity'].mean()
    merge_df.loc[i,'num_positive'] = tweets_sub.loc[tweets_sub['Class']=='positive',:].shape[0]
    merge_df.loc[i,'num_negative'] = tweets_sub.loc[tweets_sub['Class']=='negative',:].shape[0]
    merge_df.loc[i,'ratio_positive'] = merge_df.loc[i,'num_positive']/tweets_sub.shape[0]
    merge_df.loc[i,'ratio_negative'] = merge_df.loc[i,'num_negative']/tweets_sub.shape[0]

  merge_df.dropna(inplace=True)
  merge_df.set_index('Date',inplace=True)

  train_df = merge_df.loc[merge_df.index<=split_date]
  test_df = merge_df.loc[merge_df.index>split_date]
  return train_df,test_df



def get_train_test_data(train_df,test_df,step_size=60,target_size=1):
    sub_col = range(train_df.shape[1])
    train_data = train_df.iloc[:,sub_col].copy()
    test_data = test_df.iloc[:,sub_col].copy()
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    train_data = np.array(train_data)
    X_train = []
    y_train = []
    for i in range(step_size,train_data.shape[0]-target_size):
        X_train.append(train_data[i-step_size:i]) 
        y_train.append(train_data[i:i+target_size,3])
  
        
    X_train,y_train = np.array(X_train),np.array(y_train)
    
    train_data = train_df.iloc[:,sub_col].copy()
    past_step_days = train_data.tail(step_size)
    test_data = past_step_days.append(test_data,ignore_index=True)
    test_data = scaler.transform(test_data)
    test_data = np.array(test_data)
    X_test = []
    y_test = []
    for i in range(step_size,test_data.shape[0]):
        X_test.append(test_data[i-step_size:i])
        y_test.append(test_data[i,3])
    X_test,y_test = np.array(X_test),np.array(y_test)

    return X_train,y_train,X_test,y_test,scaler

def get_lstm_results(company,target_size):
    y_pred_list=[]
    y_true_list=[]
    print(company)
    ## Load Stock and Tweets Data
    stock_df = load_stock_data(company)
    tweets_df = load_tweets_df(company)
    ## Process Tweets Data
    tweets_df['Text']=tweets_df['Text'].apply(cleanText)
    tweets_df['Subjectivity']=tweets_df['Text'].apply(getSubjectivity)
    tweets_df['Polarity']=tweets_df['Text'].apply(getPolarity)
    tweets_df['Class']=tweets_df['Polarity'].apply(getClass)
    ## Merge Stock and Tweets Data
    split_date = '2021-07-31'
    train_df,test_df = mergeData(stock_df,tweets_df,split_date)
    ## Get Train and Test Data
    step_size=2
    X_train,y_train,X_test,y_test,scaler = get_train_test_data(train_df,test_df,step_size,target_size)
    X_train_x= X_train.reshape(X_train.shape[0], -1)
    X_test_x= X_test.reshape(X_test.shape[0], -1)
    json_file = open(f"./models/LSTM/"+company+"/"+company+"_"+str(target_size)+"day_model.json", 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(f"./models/LSTM/"+company+"/"+company+"_"+str(target_size)+"day_model.h5")
    prediction = model.predict(X_test )
    y_pred = model.predict(X_test)
    y_pred_df = scaler.transform(test_df)
    y_pred_df[:,3] = y_pred[:,0]
    y_pred = pd.DataFrame(scaler.inverse_transform(y_pred_df))[3]

    y_true = test_df.values[:,3]

    tester=pd.DataFrame(data=y_pred.to_list(),index=test_df.index,columns=['Predict'])
    tester.reset_index(inplace=True)
    states_buy, states_sell, total_gains, invest =buy_stock(tester.Predict, initial_state = 1,  delay = 4, initial_money = 10000)
    tester['Buy'] = np.nan
    for i in states_buy:
      tester['Buy'].iloc[i] = tester['Predict'].iloc[i]
    tester['Sell']=np.nan
    for j in states_sell:
      tester['Sell'].iloc[j]=tester['Predict'].iloc[j]    
    stocker=stock_df.drop(columns=["Open","High","Low","Adj Close","Volume"])
    merged_df=pd.merge(stocker,tester, how='outer', on="Date")
    resp_json=merged_df.to_json(orient='records',index=True)
    return resp_json


def buy_stock(
    real_movement,
    delay = 5,
    initial_state = 1,
    initial_money = 10000,
    max_buy = 1,
    max_sell = 1,
):
    """
    real_movement = actual movement in the real world
    delay = how much interval you want to delay to change our decision from buy to sell, vice versa
    initial_state = 1 is buy, 0 is sell
    initial_money = 1000, ignore what kind of currency
    max_buy = max quantity for share to buy
    max_sell = max quantity for share to sell
    """
    starting_money = initial_money
    delay_change_decision = delay
    current_decision = 0
    state = initial_state
    current_val = real_movement[0]
    states_sell = []
    states_buy = []
    current_inventory = 0

    def buy(i, initial_money, current_inventory):
        shares = initial_money // real_movement[i]
        if shares < 1:
            print(
                'day %d: total balances %f, not enough money to buy a unit price %f'
                % (i, initial_money, real_movement[i])
            )
        else:
            if shares > max_buy:
                buy_units = max_buy
            else:
                buy_units = shares
            initial_money -= buy_units * real_movement[i]
            current_inventory += buy_units
            print(
                'day %d: buy %d units at price %f, total balance %f'
                % (i, buy_units, buy_units * real_movement[i], initial_money)
            )
            states_buy.append(0)
        return initial_money, current_inventory

    if state == 1:
        initial_money, current_inventory = buy(
            0, initial_money, current_inventory
        )

    for i in range(1, real_movement.shape[0], 1):
        if real_movement[i] < current_val and state == 0:
            if current_decision < delay_change_decision:
                current_decision += 1
            else:
                state = 1
                initial_money, current_inventory = buy(
                    i, initial_money, current_inventory
                )
                current_decision = 0
                states_buy.append(i)
        if real_movement[i] > current_val and state == 1:
            if current_decision < delay_change_decision:
                current_decision += 1
            else:
                state = 0

                if current_inventory == 0:
                    print('day %d: cannot sell anything, inventory 0' % (i))
                else:
                    if current_inventory > max_sell:
                        sell_units = max_sell
                    else:
                        sell_units = current_inventory
                    current_inventory -= sell_units
                    total_sell = sell_units * real_movement[i]
                    initial_money += total_sell
                    try:
                        invest = (
                            (real_movement[i] - real_movement[states_buy[-1]])
                            / real_movement[states_buy[-1]]
                        ) * 100
                    except:
                        invest = 0
                    print(
                        'day %d, sell %d units at price %f, investment %f %%, total balance %f,'
                        % (i, sell_units, total_sell, invest, initial_money)
                    )

                current_decision = 0
                states_sell.append(i)
        current_val = real_movement[i]
    invest = ((initial_money - starting_money) / starting_money) * 100
    total_gains = initial_money - starting_money
    return states_buy, states_sell, total_gains, invest

# if __name__=="__main__":
#   get_lstm_results("AAPL",2)