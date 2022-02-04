import torch

import py_vollib
from py_vollib.black_scholes_merton import black_scholes_merton

from py_vollib.black_scholes_merton.greeks.analytical import delta
from py_vollib.black_scholes_merton.greeks.analytical import gamma
from py_vollib.black_scholes_merton.greeks.analytical import rho
from py_vollib.black_scholes_merton.greeks.analytical import theta
from py_vollib.black_scholes_merton.greeks.analytical import vega

# Auto differentiating greeks test.
# https://eadains.github.io/OptionallyBayesHugo/posts/option_pricing/#black-scholes

# Might switch from pricing library to this implementation

# params
# moneyness = 1.2
# time_to_maturity = torch.tensor(0.014, requires_grad=True) 
# dividend_rate = torch.tensor(0.1, requires_grad=True) 
# annualized_interest_rate = torch.tensor(0.1, requires_grad=True) 
# volatility = torch.tensor(0.2, requires_grad=True)

# underlyingPrice = torch.tensor(1.0, requires_grad=True)
# strike = underlyingPrice / moneyness

# # price
# bsCall = black_scholes_merton('c', underlyingPrice, strike, time_to_maturity, annualized_interest_rate, volatility, dividend_rate)
# # greeks
# bsCall_delta = delta('c', underlyingPrice, strike, time_to_maturity, annualized_interest_rate, volatility, dividend_rate)
# bsCall_rho = rho('c', underlyingPrice, strike, time_to_maturity, annualized_interest_rate, volatility, dividend_rate)
# bsCall_theta = theta('c', underlyingPrice, strike, time_to_maturity, annualized_interest_rate, volatility, dividend_rate)
# bsCall_vega = vega('c', underlyingPrice, strike, time_to_maturity, annualized_interest_rate, volatility, dividend_rate)


from torch.distributions import Normal

std_norm_cdf = Normal(0, 1).cdf
std_norm_pdf = lambda x: torch.exp(Normal(0, 1).log_prob(x))

def bs_price(right, K, S, T, sigma, r):
    d_1 = (1 / (sigma * torch.sqrt(T))) * (torch.log(S / K) + (r + (torch.square(sigma) / 2)) * T)
    d_2 = d_1 - sigma * torch.sqrt(T)
    
    if right == "C":
        C = std_norm_cdf(d_1) * S - std_norm_cdf(d_2) * K * torch.exp(-r * T)
        return C
        
    elif right == "P":
        P = std_norm_cdf(-d_2) * K * torch.exp(-r * T) - std_norm_cdf(-d_1) * S
        return P
    
right = "C"
# Strike price
K = torch.tensor(100.0, requires_grad=True)
# Underlying price
S = torch.tensor(100.0, requires_grad=True)
# total time to expiry left in years
T = torch.tensor(1.0, requires_grad=True)
# volatility
sigma = torch.tensor(0.05, requires_grad=True)
# risk free interest rate
r = torch.tensor(0.01, requires_grad=True)

price = bs_price(right, K, S, T, sigma, r)
print(price)

price.backward()
print(f"Delta: {S.grad} Vega: {sigma.grad} Theta: {T.grad} Rho: {r.grad}")

# print(f'Call price: {bsCall}.')

# print(f'Delta: {bsCall_delta}. Rho: {bsCall_rho}. Theta: {bsCall_theta}. Vega: {bsCall_vega}.')

