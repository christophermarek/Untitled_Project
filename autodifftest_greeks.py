from math import sqrt
from subprocess import ABOVE_NORMAL_PRIORITY_CLASS
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal
import random
from torch import nn, optim

num_epoches = 1000

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
x_vals = []
prices = []
greeks = []


for i in range(num_epoches):
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
    x_vals.append(torch.tensor([K.item(),S.item(),T.item(),sigma.item(),r.item()], dtype=torch.float32))
    prices.append(price.item())
    greeks.append(np.array([K.item(), S.grad.item(), sigma.grad.item(), T.grad.item(), r.grad.item()]))


print('done generating data')

model = torch.nn.Sequential(
    torch.nn.Linear(5, 32),
    torch.nn.Sigmoid(),
    torch.nn.Linear(32, 1)
)


# print(x_vals[2])
lossfn = nn.MSELoss()

opt = torch.optim.Adam(model.parameters(), lr=0.1)
for n in range(num_epoches):
    opt.zero_grad()
    model_out = model(x_vals[n].float())
    train_y = torch.tensor([prices[n]])
    loss = lossfn(model_out, train_y)
    print(loss.item())
    loss.backward()
    opt.step()

torch.save(model.state_dict(), 'models/autodifftest.ckpt')

# CANT I MAKE A PLOT OF PRICE, strike, underlying, delta

# ok so actually visualize model output, maybe residual plot first then 
# auto diff check to make sure im not wasting time, then residual plot on deltas,
# then residual plot on what i said above

# should i plot strike and underlying? or just moneyness, idk

pred_price = []
pred_greeks = []

delta = []
pred_delta = []

theta = []
pred_theta = []

vega = []
pred_vega = []

rho = []
pred_rho = []

for i in range(num_epoches):
    # PROBABLY HAVE TO LOOP HERE TO TRACK AUTODIFF PROPERLY
    x_vals[i].requires_grad = True
    z = model(x_vals[i])
    pred_price.append(z.item())
    # Maybe itll work if i just dont do the sum? hope so!!!
    # or else what do i do?!
    z.backward()
    pred_greeks.append(x_vals[i].grad)
    
    delta.append(greeks[i][1])
    theta.append(greeks[i][2])
    vega.append(greeks[i][3])
    rho.append(greeks[i][4])
    
    pred_delta.append(pred_greeks[i][1])
    pred_theta.append(pred_greeks[i][2])
    pred_vega.append(pred_greeks[i][3])
    pred_rho.append(pred_greeks[i][4])
    
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)
 


plt.clf()
plt.scatter(prices, pred_price, marker=".", label="price residual")
plt.xlabel('Test call price')
plt.ylabel('Pred call price')
plt.title('MSE: ' + str(rmse_metric(prices, pred_price)))
plt.savefig('test/Call2.png', bbox_inches="tight")
plt.clf()
plt.scatter(delta, pred_delta, marker=".", label="delta residual")
plt.xlabel('Test delta')
plt.ylabel('Pred delta')
plt.title('MSE: ' + str(rmse_metric(delta, pred_delta)))
plt.savefig('test/delta2.png', bbox_inches="tight")
plt.clf()
plt.scatter(theta, pred_theta, marker=".", label="theta residual")
plt.xlabel('Test theta')
plt.ylabel('Pred theta')
plt.title('MSE: ' + str(rmse_metric(theta, pred_theta)))
plt.savefig('test/theta2.png', bbox_inches="tight")
plt.clf()
plt.scatter(vega, pred_vega, marker=".", label="vega residual")
plt.xlabel('Test vega')
plt.ylabel('Pred vega')
plt.title('MSE: ' + str(rmse_metric(vega, pred_vega)))
plt.savefig('test/vega2.png', bbox_inches="tight")
plt.clf()
plt.scatter(rho, pred_rho, marker=".", label="rho residual")
plt.xlabel('Test rho')
plt.ylabel('Pred rho')
plt.title('MSE: ' + str(rmse_metric(rho, pred_rho)))
plt.savefig('test/rho2.png', bbox_inches="tight")
plt.clf()
    

# with torch.no_grad():
#     ax2.plot(x.numpy(), x.grad.numpy(), label='model deriv')

# plot price residual
# PLOT WITH MSE or loss function valueb


# greeks residuals (4 of them)

# TITLE EACH GREEK PLOT WITH THE MSE FOR THAT GREEK



# plot of moneyness, delta, price, pred price



# ax1.legend()
# ax2.legend()
# plt.show()