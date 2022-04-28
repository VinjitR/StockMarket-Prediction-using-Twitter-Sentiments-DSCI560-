import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMAResults
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import pickle


root_dir='./data/'
def train_test_split(timeseries, split_date ='2021-07-31' ):
  ts = timeseries.copy()

  train_df = ts.loc[ts.index <= split_date]
  test_df = ts.loc[ts.index > split_date]
  return train_df,test_df

def load_stock_data(company):
  path = root_dir
  stock_filename=path+company+'.csv'
  stock_df=pd.read_csv(stock_filename)
  stock_df['Date']=pd.to_datetime(stock_df['Date'])
  stock_df=stock_df[stock_df['Date']>='2019-01-01']
  stock_df=stock_df[stock_df['Date']<='2021-12-31']

  return stock_df

def extract_hist_price(stock_df):
  price_data = stock_df[['Date','Close']]
  price_data.Date = pd.to_datetime(price_data.Date)
  price_data = price_data.set_index('Date')
  return price_data  

def get_stationarity(timeseries):
    
    # rolling statistics
    rolling_mean = timeseries.rolling(window=5).mean()
    rolling_std = timeseries.rolling(window=5).std()
    
    # rolling statistics plot
    # original = plt.plot(timeseries, color='blue', label='Original')
    # mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    # std = plt.plot(rolling_std, color='black', label='Rolling Std')
    # plt.legend(loc='best')
    # plt.title('Rolling Mean & Standard Deviation')
    # plt.show(block=False)
    
    # Dickey–Fuller test:
    result = adfuller(timeseries['Close'])
    # print('ADF Statistic: {}'.format(result[0]))
    # print('p-value: {}'.format(result[1]))
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print('\t{}: {}'.format(key, value))


def get_results(company):
    print(company)
    ## Load Stock and Tweets Data
    stock_df = load_stock_data(company)
    stock_price = extract_hist_price(stock_df)


    price_log = np.log(stock_price)  
    price_log_shift = price_log - price_log.shift()
    price_log_shift.dropna(inplace=True)
    print(get_stationarity(price_log_shift))

    split_date = '2021-7-31'

    # train_df,test_df = train_test_split(price_log_exp_decay)
    train_df,test_df = train_test_split(price_log_shift)
    loaded = ARIMAResults.load('./ARIMA_new/1day/arima_model_1day_'+company+'.pkl')
    print(loaded)
    # start=len(train_df)
    # end=len(train_df)+len(test_df)-1

    # pred = pickle_preds.predict(start=start,end=end).rename('1-day ARIMA Predictions')
    shift = price_log.shift()
    shift_log_test = shift.loc[shift.index > split_date]

    ad = np.array(shift_log_test.Close)
    pred_arr = np.array(pred)
    sum_two = (pred_arr + ad) if ad != 'nan' else pred_arr
    price_pred = np.exp(sum_two)
    return price_pred,test_df.index
    



# company_list=['AAPL','AMZN','FB','GOOGL','MSFT','TSLA']

# for company in company_list:
#     print(company)
#     ## Load Stock and Tweets Data
#     stock_df = load_stock_data(company)
#     stock_price = extract_hist_price(stock_df)


#     price_log = np.log(stock_price)  
#     price_log_shift = price_log - price_log.shift()
#     price_log_shift.dropna(inplace=True)
#     print(get_stationarity(price_log_shift))

#     split_date = '2021-7-31'

#     # train_df,test_df = train_test_split(price_log_exp_decay)
#     train_df,test_df = train_test_split(price_log_shift)
#     stepwise_fit = auto_arima(price_log_shift['Close'], trace=True,suppress_warnings=True)
#     # print(stepwise_fit)

#     # model=ARIMA(price_log_exp_decay['Close'],order=(1,0,0))
#     model=ARIMA(price_log_shift['Close'],order=(2,0,0))
#     model=model.fit()
#     model.summary()


    
#     # create dir & save model
#     path ='/content/drive/MyDrive/560 Project/ARIMA_new/1day/'
#     # dir_name = "arima_model_{}".format(company)

#     # if not os.path.exists(dir_name): os.mkdir(dir_name)
    
#     pkl_filename = "arima_model_1day:{}.pkl".format(company)
#     full_path = path + pkl_filename
#     print(full_path)
#     pickle.dump(model, open(full_path, 'wb'))





#     start=len(train_df)
#     end=len(train_df)+len(test_df)-1
#     pred = model.predict(start=start,end=end).rename('1-day ARIMA Predictions')

#     shift = price_log.shift()
#     shift_log_test = shift.loc[shift.index > split_date]

#     ad = np.array(shift_log_test.Close)
#     pred_arr = np.array(pred)
#     sum_two = (pred_arr + ad) if ad != 'nan' else pred_arr
#     price_pred = np.exp(sum_two)

#     # true stock price
#     true_test = stock_price[stock_price.index > split_date].Close


#     test_df.mean()
#     mse = mean_squared_error(price_pred,true_test)
#     print(company ,':', mse)

#     plt.plot(test_df.index, price_pred)
#     plt.plot(test_df.index, true_test)
#     plt.title('1-Day Stock Price Predictions for '+company)
#     plt.legend()
#     plt.show()