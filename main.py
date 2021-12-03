
from numpy import character, number
import py_vollib
from py_vollib.black_scholes_merton import black_scholes_merton
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from py_vollib.black_scholes_merton.greeks.analytical import delta
from py_vollib.black_scholes_merton.greeks.analytical import gamma
from py_vollib.black_scholes_merton.greeks.analytical import rho
from py_vollib.black_scholes_merton.greeks.analytical import theta
from py_vollib.black_scholes_merton.greeks.analytical import vega
from datetime import datetime, date
import numpy as np
import pandas as pd
# import pandas_datareader.data as web
import bs_calc

import data_generator

def pyvolib(S, K, sigma, r, flag: character, t, q):
    #IV and prices

    # Calculate the Black-Scholes-Merton implied volatility.
    price = black_scholes_merton(flag, S, K, t, r, sigma, q)
    iv = implied_volatility(price, S, K, t, r, q, flag)
    # Return the Black-Scholes-Merton option price.
    p_calc = black_scholes_merton('p', S, K, t, r, sigma, q)

    # Greeks
    delta_calc = delta(flag, S, K, t, r, sigma, q)
    gamma_calc = gamma(flag, S, K, t, r, sigma, q)
    rho_calc = rho(flag, S, K, t, r, sigma, q)
    # The text book analytical formula does not divide by 365, but in practice theta is defined as the change in price for each day change in t, hence we divide by 365
    annual_theta_calc = theta(flag, S, K, t, r, sigma, q) * 365
    vega_calc = vega(flag, S, K, t, r, sigma, q)

    return delta_calc, gamma_calc, rho_calc, annual_theta_calc, vega_calc

def getYahooStockPrices():
    

    stock = 'SPY'
    expiry = '12-18-2022'
    strike_price = 370

    today = datetime.now()
    one_year_ago = today.replace(year=today.year-1)

    df = web.DataReader(stock, 'yahoo', one_year_ago, today)

    df = df.sort_values(by="Date")
    df = df.dropna()
    df = df.assign(close_day_before=df.Close.shift(1))
    df['returns'] = ((df.Close - df.close_day_before)/df.close_day_before)

    sigma = np.sqrt(252) * df['returns'].std()
    uty = (web.DataReader(
        "^TNX", 'yahoo', today.replace(day=today.day-1), today)['Close'].iloc[-1])/100
    lcp = df['Close'].iloc[-1]
    t = (datetime.strptime(expiry, "%m-%d-%Y") - datetime.utcnow()).days / 365

    print('The Option Price is: ', bs_call(lcp, strike_price, t, uty, sigma))

def main():
    print("Begin Comparison")
    # S (float) – underlying asset price
    # K (float) – strike price
    # sigma (float) – annualized standard deviation, or volatility
    # t (float) – time to expiration in years
    # r (float) – risk-free interest rate
    # q (float) – annualized continuous dividend rate
    # flag (str) – ‘c’ or ‘p’ for call or put.

    S = 100
    K = 100
    sigma = .2
    r = .01
    flag = 'c'
    t = .5
    q = 0

    print(pyvolib(S, K, sigma, r, flag, t, q))
    print(bs_calc.Black_Scholes_Greeks_Call(S, K, r, sigma, t, 0))
    #They are identical
    print("Comparison complete")
    
    # So obv just use pyvolib instead of bs_calc, but bs_calc i can modify if I want which could be useful

    #now using actual data
    # i have the function but I need to make it actually return something now.
    # Well it wil return the stock price 
    # and we will put it into our function to calculate the greeks
    # then we will compare it to the actual option greek values at that time.
    #The link below shows where to get the real world data
    #https://medium.com/swlh/calculating-option-premiums-using-the-black-scholes-model-in-python-e9ed227afbee

# main()

def generateData():
    dataGenerator = data_generator.DataGenerator()
    
    destination = "demo_file.csv"
    output = dataGenerator.generateDataSet(1000, destination)
    
    print(output[1])
    if not output[0]:
        return
    
    

    # Then output table of data showing few rows
    df = pd.read_csv(destination)
    print(df.head()) 
    
    # Then output scatterplot of data generated
    # prob should have an output folder with the file, and figures

generateData()