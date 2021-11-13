
import py_vollib
from py_vollib.black_scholes_merton import black_scholes_merton
from py_vollib.black_scholes_merton import implied_volatility


def main():

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

    # Calculate the Black-Scholes-Merton implied volatility.
    price = black_scholes_merton(flag, S, K, t, r, sigma, q)
    iv = implied_volatility(price, S, K, t, r, q, flag)


    # Return the Black-Scholes-Merton option price.
    p_calc = black_scholes_merton('p', S, K, t, r, sigma, q)


    print(price)

main()
