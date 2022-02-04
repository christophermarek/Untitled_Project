import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal
import random
import tensorflow as tf

# x = torch.linspace(-1, 1, 100).unsqueeze(1)
# y_prime = 3 * x**2 + torch.normal(torch.zeros(100,1), torch.ones(100,1) * 0.1)
# y = y_prime / 3 * x


# GENERATE DATASET
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

# same index = same dataset
x_vals = np.empty((1000, 5))
prices = np.empty((1000, 1))
greeks = np.empty((1000, 4))


for i in range(1000):
    right = "C"
    moneyness = random.uniform(0.8, 1.2)
    underlyingPrice = 1.0
    strike = underlyingPrice / moneyness
    
    
    # Strike price
    K = torch.tensor(strike, requires_grad=True)
    # Underlying price
    S = torch.tensor(underlyingPrice, requires_grad=True)
    # total time to expiry left in years
    T = torch.tensor(random.uniform(0.014, 1), requires_grad=True)
    # volatility
    sigma = torch.tensor(random.uniform(0.1, 0.4), requires_grad=True)
    # risk free interest rate
    r = torch.tensor(random.uniform(0.00, 0.1), requires_grad=True)
    
    price = bs_price(right, K, S, T, sigma, r)

    price.backward()
    # print(f"Delta: {S.grad} Vega: {sigma.grad} Theta: {T.grad} Rho: {r.grad}")
    x_vals[i] = np.array([K.item(),S.item(),T.item(),sigma.item(),r.item()])
    prices[i] = price.item()
    greeks[i] = np.array([S.grad.item(), sigma.grad.item(), T.grad.item(), r.grad.item()])


print('done generating data')

model = torch.nn.Sequential(
    torch.nn.Linear(5, 32),
    torch.nn.Sigmoid(),
    torch.nn.Linear(32, 1)
)


opt = torch.optim.Adam(model.parameters(), lr=0.1)
for n in range(1000):
    opt.zero_grad()
    loss = torch.mean((model(tf.convert_to_tensor(x_vals[n], np.float32)) - tf.convert_to_tensor(price[n], np.float32))**2)
    print(loss)
    loss.backward()
    opt.step()

# with torch.no_grad():
#     fig, (ax1, ax2) = plt.subplots(2, 1, num=1)
#     ax1.plot(x.numpy(), y.numpy(), '.', label='data')
#     ax1.plot(x.numpy(), model(x).numpy(), label='model')
#     ax2.plot(x.numpy(), y_prime.numpy(), '.', label='data deriv')

# x.requires_grad = True
# z = model(x)
# z.sum().backward()

# with torch.no_grad():
#     ax2.plot(x.numpy(), x.grad.numpy(), label='model deriv')

# ax1.legend()
# ax2.legend()
# plt.show()