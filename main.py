
from numpy import character, number
import py_vollib
from py_vollib.black_scholes_merton import black_scholes_merton
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from py_vollib.black_scholes_merton.greeks.analytical import delta
from py_vollib.black_scholes_merton.greeks.analytical import gamma
from py_vollib.black_scholes_merton.greeks.analytical import rho
from py_vollib.black_scholes_merton.greeks.analytical import theta
from py_vollib.black_scholes_merton.greeks.analytical import vega

import bs_calc

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

main()
