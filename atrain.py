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
from stable_baselines3.common.vec_env import VecCheckNan

from pprint import pprint
import sys
import itertools
import os
import argparse

today = datetime.datetime.today()
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-m', '--model', type=str, default="sac", help='model')
parser.add_argument('-d', '--data_dir', type=str, default="data", help='data_dir')
parser.add_argument('-g', '--gpu', type=str, default="", help='gpu')

ARGS = parser.parse_args()



os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.gpu


processed_full = pd.read_csv("data/datasets/processed_sz50.csv")
train = data_split(processed_full, '2009-01-01','2019-01-01')
trade = data_split(processed_full, '2019-01-01','2021-01-01')
print(len(train))
print(len(trade))
stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")




data_turbulence = processed_full[(processed_full.date<'2019-01-01') & (processed_full.date>='2009-01-01')]
insample_turbulence = data_turbulence.drop_duplicates(subset=['date'])
turbulence_threshold = np.quantile(insample_turbulence.turbulence.values,1)
print("turbulence_threshold:", turbulence_threshold)






env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4

}

e_train_gym = StockTradingEnv(df = train, **env_kwargs)

env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))
agent = DRLAgent(env = env_train)


def get_model(model_name):
    if model_name == "s2c" or model_name == "ddpg":
        model = agent.get_model(model_name)
    elif model_name == "ppo":
        PPO_PARAMS = {
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 128,
        }
        model = agent.get_model(model_name, model_kwargs = PPO_PARAMS)
    elif model_name == 'td3':
        TD3_PARAMS = {"batch_size": 100,
              "buffer_size": 1000000,
              "learning_rate": 0.001}
        model = agent.get_model("td3",model_kwargs = TD3_PARAMS)
    elif model_name == 'sac':
        SAC_PARAMS = {
            "batch_size": 128,
            "buffer_size": 1000000,
            "learning_rate": 0.0001,
            "learning_starts": 100,
            "ent_coef": "auto_0.1",
        }

        model = agent.get_model("sac",model_kwargs = SAC_PARAMS)
    return model


model_path = "./data/trained_models/ast"
def train(model_name, total_timesteps):
    model = get_model(model_name)
    trained = agent.train_model(model=model,
                             tb_log_name=model_name,
                             total_timesteps=total_timesteps)
    trained.save(model_path + model_name)
    return trained

def load(model_name):
    return agent.load_model(model_name,model_path+ model_name)


def load_or_train(model_name, total_timesteps = 30000):
    if os.path.exists(model_path + model_name + ".zip"):
        print("load " + model_name)
        return load(model_name)
    else:
        return train(model_name, total_timesteps)

#train("a2c", 100000)
#train("ddpg", 50000)
#train("ppo", 50000)
#train("td3", 30000)
#train("sac", 80000)


# train model
#train("td3", 10)
#m = load_or_train("ddpg", 10)
#m = load_or_train("ppo", 50000)
#m = load_or_train("ddpg", 50000)
m = load_or_train(ARGS.model, 80000)
print(m)


# simulate for trade
e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = turbulence_threshold, **env_kwargs)
df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=m,
    environment = e_trade_gym)

print(df_account_value)
print(df_actions)




print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')


#print("==============Get Baseline Stats===========")
#baseline_df = get_baseline(
#        ticker="^DJI",
#        start = '2019-01-01',
#        end = '2021-01-01')
#
#stats = backtest_stats(baseline_df, value_col_name = 'close')
#
#
#
#
#print("==============Compare to DJIA===========")
## S&P 500: ^GSPC
## Dow Jones Index: ^DJI
## NASDAQ 100: ^NDX
#backtest_plot(df_account_value,
#             baseline_ticker = '^DJI',
#             baseline_start = '2019-01-01',
#             baseline_end = '2021-01-01')

