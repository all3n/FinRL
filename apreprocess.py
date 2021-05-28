import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline

from pprint import pprint
import sys
import itertools
import argparse

today = datetime.datetime.today()
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-j', '--job', type=str, default="train", help='job')
parser.add_argument('-d', '--data_dir', type=str, default="data", help='data_dir')

ARGS = parser.parse_args()




def day_of_week(date):
    d = datetime.datetime.strptime(date, "%Y-%m-%d")
    return d.weekday()


df = pd.read_csv(ARGS.data_dir + '/input.csv')

#df['date'] = df.index
df['day'] = df['date'].map(day_of_week)
df.rename(columns = {"code": "tic"},  inplace=True)



fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                    use_turbulence=True,
                    user_defined_feature = False)

processed = fe.preprocess_data(df)

list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
combination = list(itertools.product(list_date,list_ticker))

processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date','tic'])

processed_full = processed_full.fillna(0)

processed_full.to_csv(ARGS.data_dir + "/output.csv", index = False)
