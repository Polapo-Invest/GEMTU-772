# Import Packages
import numpy as np
from scipy.optimize import minimize
import yfinance as yf


def get_etf_price_data():
    tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOG', 'XLK']
    etf = yf.Tickers(tickers)
    data = etf.history(start='2010-01-01', actions=False)
    data.drop(['Open', 'High', 'Low', 'Volume'], inplace=True, axis=1)
    data = data.droplevel(0, axis=1) # Change line 16 to data = data.droplevel(0, axis=1).resample('M').last() if you want it to be monthly. Current code is based on weekly
    data.ffill(inplace=True)
    df = data.resample('W').last()
    return df

df = get_etf_price_data()

print(df)