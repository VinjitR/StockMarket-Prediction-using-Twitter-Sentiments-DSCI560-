
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import re 
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler

company_list=['AAPL','AMZN','FB','GOOGL','MSFT','TSLA']
root_dir="./data/"

def load_stock_data(company, path = root_dir):
  stock_filename=path + company +'.csv'
  stock_df=pd.read_csv(stock_filename)
  stock_df['Date']=pd.to_datetime(stock_df['Date'])
  stock_df=stock_df[stock_df['Date']>='2019-01-01']
  stock_df=stock_df[stock_df['Date']<='2021-12-31']

  return stock_df

def load_tweets_df(company, path = root_dir):
  tweets_file_name=path+company+'_tweets.csv'
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


def mergeData(stock_df,tweets_df):
  merge_df = stock_df.copy()
  for i in range(0,len(stock_df)):
    cur_time = stock_df.iloc[i,:]['Date']
    idx = tweets_df['Datetime'] == cur_time
    tweets_sub = tweets_df.loc[idx]

    tweets_sub['len_chars'] = tweets_sub['Text'].apply(len)
    merge_df.loc[i,'len_chars'] = tweets_sub['len_chars'].mean()
    merge_df.loc[i,'polarity'] = tweets_sub['Polarity'].mean()


  merge_df.dropna(inplace=True)
  merge_df.set_index('Date',inplace=True)


  return merge_df


def daily_polarity(company):
  stock_df = load_stock_data(company)
  tweets_df = load_tweets_df(company)
  tweets_df['Text']=tweets_df['Text'].apply(cleanText)
  tweets_df['Subjectivity']=tweets_df['Text'].apply(getSubjectivity)
  tweets_df['Polarity']=tweets_df['Text'].apply(getPolarity)
  # merged_df = mergeData(stock_df,tweets_df)
  tweets_df['Class']=tweets_df['Polarity'].apply(getClass)
  return tweets_df

def get_tweets_sentiment(company,date):
  tweets_df=daily_polarity(company)
  return tweets_df[(tweets_df['Datetime'] == date)]["Text"].to_list()

def get_tweets_percentages(company):
  tweets_df=daily_polarity(company)
  class_list=tweets_df["Class"].to_list()
  positives,negatives,neutrals=class_list.count("positive"),class_list.count("negative"),class_list.count("neutral")
  positive_rate=(positives/len(class_list))*100
  negative_rate=(negatives/len(class_list))*100
  neutral_rate=(neutrals/len(class_list))*100
  # rate_dict={"pr":positive_rate,"nr":negative_rate,"neu":neutral_rate}
  rate=[{"name":"positivity rate","value":positive_rate},{"name":"negativity rate","value":negative_rate},{"name":"nuetral rate","value":neutral_rate}]
  return rate

# print(get_tweets_percentages("AAPL"))