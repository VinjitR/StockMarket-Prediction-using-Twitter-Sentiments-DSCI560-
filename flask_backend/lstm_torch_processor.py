from os import O_TRUNC
import re
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
import json
from textblob import TextBlob
import sys
#============================[load & process csv data]======================================
def load_stock_data(company):
  path = './data/'
  stock_filename=path+company+'.csv'
  stock_df=pd.read_csv(stock_filename)
  stock_df['Date']=pd.to_datetime(stock_df['Date'])
  stock_df=stock_df[stock_df['Date']>='2019-01-01']
  stock_df=stock_df[stock_df['Date']<='2021-12-31']

  return stock_df


def load_tweets_df(company):
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

    path = './data/'
    tweets_file_name=path+company+'_tweets.csv'
    tweets_df=pd.read_csv(tweets_file_name,usecols=range(1,5))
    tweets_df.dropna(inplace=True)
    tweets_df['Datetime']=tweets_df['Datetime'].str.split(' ').str.get(0)
    tweets_df['Datetime']=pd.to_datetime(tweets_df['Datetime'])


    tweets_df['Text']=tweets_df['Text'].apply(cleanText)
    tweets_df['Subjectivity']=tweets_df['Text'].apply(getSubjectivity)
    tweets_df['Polarity']=tweets_df['Text'].apply(getPolarity)
    tweets_df['Class']=tweets_df['Polarity'].apply(getClass)

    return tweets_df

def mergeData(stock_df,tweets_df,split_date='2021-07-31'):
    merge_df = stock_df.copy()

    for i in range(0,len(stock_df)):
        cur_time = stock_df.iloc[i,:]['Date']
        idx = tweets_df['Datetime'] == cur_time
        tweets_sub = tweets_df.loc[idx]

        merge_df.loc[i,'polarity'] = tweets_sub['Polarity'].mean()
        merge_df.loc[i,'subjectivity'] = tweets_sub['Subjectivity'].mean()
        merge_df.loc[i,'num_positive'] = tweets_sub.loc[tweets_sub['Class']=='positive',:].shape[0]
        merge_df.loc[i,'num_negative'] = tweets_sub.loc[tweets_sub['Class']=='negative',:].shape[0]
        merge_df.loc[i,'ratio_positive'] = merge_df.loc[i,'num_positive']/tweets_sub.shape[0]
        merge_df.loc[i,'ratio_negative'] = merge_df.loc[i,'num_negative']/tweets_sub.shape[0]

    merge_df.dropna(inplace=True)
    # merge_df.set_index('Date',inplace=True)

    train_df = merge_df.loc[merge_df['Date']<=split_date]
    test_df = merge_df.loc[merge_df['Date']>split_date]
    return train_df,test_df

#=================================[dataset]=====================================
class Dataset(data.Dataset):
    def __init__(self, data_df, input_steps):
        self.X = torch.from_numpy(pd.DataFrame(data_df, columns=['Open','High','Low','Close','polarity','subjectivity','ratio_positive','ratio_negative']).values)
        self.Y = torch.from_numpy(pd.DataFrame(data_df, columns=['Close',]).values)
        self.input_steps = input_steps
        
    def __len__(self):
        return self.X.shape[0]-3

    def __getitem__(self, index):
        x = self.X[index+1-self.input_steps if index>self.input_steps else 0:index+1, :]
        y = self.Y[index+1:index+3+1]

        return x, y

#================================[model]========================================


class LSTMs(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout_rate):
        super(LSTMs, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers = self.num_layers, 
                            dropout = dropout_rate, batch_first = True)
        
        self.output_layer = nn.Linear(self.hidden_dim, output_dim)
        
    def forward(self, x):
        output, _ = self.lstm(x.to(torch.float32)) 
        output = self.output_layer(output[:,-1:,:].float())
        return output

model=LSTMs(8,3,64,1,0.2)
def predict_torch(company,target):

    # paths={
    # "AAPL":"./save_and_load/lstm_model_AAPL/LSTM_AAPL_5.pkl",
    # "AMZN":"./save_and_load/lstm_model_AMZN/LSTM_AMZN_2.pkl",
    # "FB":"./save_and_load/lstm_model_FB/LSTM_FB_2.pkl",
    # "GOOGL":"./save_and_load/lstm_model_GOOGL/LSTM_GOOGL_5.pkl",
    # "MSFT":"./save_and_load/lstm_model_MSFT/LSTM_MSFT_2.pkl",
    # "TSLA":"./save_and_load/lstm_model_TSLA/LSTM_TSLA_10.pkl"
    # }
    paths={"AAPL":"./lstm_check/AAPL.pkl"}
    model=torch.load(paths[company])
    model.eval()
    model.to("cpu")
    # torch.device("cpu")
    # model = torch.load("")
    print(model)
    # load stock data
    stock_df    = load_stock_data(company)
    # load tweets data
    tweets_df   = load_tweets_df(company)
    # merge and split data
    _,test_df = mergeData(stock_df,tweets_df,'2021-7-31')
    # test dataset
    test_set = Dataset(test_df,30)
    print(test_df[["Date","Close"]])
    # make prediction
    predictions=[]
    for x, y in data.DataLoader(test_set):
        predictions.append(model(x).cpu().detach().numpy().tolist())
    res=[]
    for i in predictions:
        res.append(i[0][0][target])
    res.append(predictions[-1][0][0][target])
    res.append(predictions[-1][0][0][target])
    res.append(predictions[-1][0][0][target])
    print(res)
    output_df=test_df[["Date","Close"]]
    output_df["Predict"]=res
    return(output_df.to_json(orient='records',index=True))


   
if __name__=="__main__":
    com=sys.argv[1]
    out=sys.argv[2]
    target=int(sys.argv[3])
    # AAPL=manage(predict_torch("AAPL"))
    # AMZN=manage(predict_torch("AMZN"))
    # FB=manage(predict_torch("FB"))
    # MSFT=manage(predict_torch("MSFT"))
    # GOOGL=manage(predict_torch("GOOGL"))
    # TSLA=manage(predict_torch("TSLA"))
    output={com:predict_torch(com,target)}
    with open(out,'w')as fp:
        json.dump(output,fp)
    # torch_output={"AAPL":AAPL,"AMZN":AMZN,"FB":FB,"MSFT":MSFT,"GOOGL":GOOGL,"TSLA":TSLA}
    # print(torch_output)