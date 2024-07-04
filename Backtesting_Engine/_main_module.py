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

# Portfolio Backtesting Engine Class
class GEMTU772:
    # Initialization Function
    def __init__(self, price, param=52):

        # Annualization Parameter
        self.param = param

        # Intraday Return Rate
        self.rets = price.pct_change().dropna()

        # Expected Rate of Return
        self.er = np.array(self.rets * self.param)

        # Volatility
        self.vol = np.array(self.rets.rolling(self.param).std() * np.sqrt(self.param))

        # Covariance Matrix
        cov = self.rets.rolling(self.param).cov().dropna() * self.param
        self.cov = cov.values.reshape(int(cov.shape[0]/cov.shape[1]), cov.shape[1], cov.shape[1])

    # Cross-Sectional Risk Model Class 
    class CrossSectional:

        # EW
        def ew(self, er):
            noa = er.shape[0]
            weights = np.ones_like(er) * (1/noa)
            return weights
        
        # MSR
        def msr(self, er, cov):
            noa = er.shape[0]
            init_guess = np.repeat(1/noa, noa)

            bounds = ((0.0, 1.0), ) * noa
            weights_sum_to_1 = {'type': 'eq',
                                'fun': lambda weights: np.sum(weights) - 1}

            def neg_sharpe(weights, er, cov):
                r = weights.T @ er # @ means multiplication
                vol = np.sqrt(weights.T @ cov @ weights)
                return - r / vol

            weights = minimize(neg_sharpe,
                            init_guess,
                            args=(er, cov),
                            method='SLSQP',
                            constraints=(weights_sum_to_1,), 
                            bounds=bounds)

            return weights.x
        
        # GMV
        def gmv(self, cov):
            noa = cov.shape[0]
            init_guess = np.repeat(1/noa, noa)

            bounds = ((0.0, 1.0), ) * noa
            weights_sum_to_1 = {'type': 'eq',
                                'fun': lambda weights: np.sum(weights) - 1}

            def port_vol(weights, cov):
                vol = np.sqrt(weights.T @ cov @ weights)
                return vol

            weights = minimize(port_vol, init_guess, args=(cov), method='SLSQP', constraints=(weights_sum_to_1,), bounds=bounds)

            return weights.x
        
        # MDP
        def mdp(self, vol, cov):
            noa = vol.shape[0]
            init_guess = np.repeat(1/noa, noa)
            bounds = ((0.0, 1.0), ) * noa
            
            weights_sum_to_1 = {'type': 'eq',
                                'fun': lambda weights: np.sum(weights) - 1}
            
            def neg_div_ratio(weights, vol, cov):
                weighted_vol = weights.T @ vol
                port_vol = np.sqrt(weights.T @ cov @ weights)
                return - weighted_vol / port_vol
            
            weights = minimize(neg_div_ratio, 
                               init_guess, 
                               args=(vol, cov),
                               method='SLSQP',
                               constraints=(weights_sum_to_1,), 
                               bounds=bounds)
            
            return weights.x
        
        # RP
        def rp(self, cov):
            noa = cov.shape[0]
            init_guess = np.repeat(1/noa, noa)
            bounds = ((0.0, 1.0), ) * noa
            target_risk = np.repeat(1/noa, noa)
            
            weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1}
            
            def msd_risk(weights, target_risk, cov):
                
                port_var = weights.T @ cov @ weights
                marginal_contribs = cov @ weights
                
                risk_contribs = np.multiply(marginal_contribs, weights.T) / port_var
                
                w_contribs = risk_contribs
                return ((w_contribs - target_risk)**2).sum()
            
            weights = minimize(msd_risk, 
                               init_guess,
                               args=(target_risk, cov), 
                               method='SLSQP',
                               constraints=(weights_sum_to_1,),
                               bounds=bounds)
            return weights.x
        
        # EMV
        def emv(self, vol):
            inv_vol = 1 / vol
            weights = inv_vol / inv_vol.sum()
    
            return weights
        
