
#  we randomly generate the validation set after, since it is all random data anyways.
from math import sqrt
import random
from matplotlib import pyplot as plt
import numpy as np

import torch
from torch.distributions import Normal

std_norm_cdf = Normal(0, 1).cdf
def std_norm_pdf(x): return torch.exp(Normal(0, 1).log_prob(x))


def bs_price(right, K, S, T, sigma, r):
    d_1 = (1 / (sigma * torch.sqrt(T))) * \
          (torch.log(S / K) + (r + (torch.square(sigma) / 2)) * T)
    d_2 = d_1 - sigma * torch.sqrt(T)

    if right == "C":
        C = std_norm_cdf(d_1) * S - std_norm_cdf(d_2) * \
        K * torch.exp(-r * T)
        return C

    elif right == "P":
        P = std_norm_cdf(-d_2) * K * torch.exp(-r * T) - \
            std_norm_cdf(-d_1) * S
        return P

def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


model = torch.nn.Sequential(
    torch.nn.Linear(5, 400),
    torch.nn.SiLU(),
    torch.nn.Linear(400, 400),
    torch.nn.SiLU(),
    torch.nn.Linear(400, 400),
    torch.nn.SiLU(),
    torch.nn.Linear(400, 1),
    torch.nn.SiLU()
)

val_sample_size = 100000
model.load_state_dict(torch.load('done+2000000.ckpt'))


val_x_vals = []
val_prices = []
val_greeks = []
# generate validation set
# 20% size of test set
print('generating real world validation set')
for i in range(val_sample_size):
    right = "C"

    # Strike price
    K = torch.tensor(random.uniform(90, 100), requires_grad=True)
        
    # Underlying price Delta: {S.grad}
    S = torch.tensor(random.uniform(90, 100), requires_grad=True)
    # total time to expiry left in years Theta: {T.grad}
    T = torch.tensor(random.uniform(0.014, 1), requires_grad=True)
    # volatility "Vega: {sigma.grad}
    sigma = torch.tensor(random.uniform(0.1, 0.4), requires_grad=True)
    # risk free interest rate Rho: {r.grad}
    r = torch.tensor(random.uniform(0.00, 0.1), requires_grad=True)
    price = bs_price(right, K, S, T, sigma, r)

    price.backward()
            
    val_x_vals.append(torch.tensor(
    [K.item(), S.item(), T.item(), sigma.item(), r.item()], dtype=torch.float32))
    val_prices.append(price.item())
            
    # "strike,underlying,maturity,volatility,interestrate,call_price,delta,theta,vega,rho\n")
    val_greeks.append(np.array([S.grad.item(), T.grad.item(), sigma.grad.item(), r.grad.item()]))
    
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

moneyness = []
strike = []
timetomaturity = []

# validate
for i in range(val_sample_size):
    val_x_vals[i].requires_grad = True
    # print(x_vals[i])
    z = model(val_x_vals[i])
    pred_price.append(z.item())
    z.sum().backward()
    pred_greeks.append(val_x_vals[i].grad)
    delta.append(val_greeks[i][0])
    theta.append(val_greeks[i][1])
    vega.append(val_greeks[i][2])
    rho.append(val_greeks[i][3])

for i in range(len(pred_greeks)):
    pred_delta.append(pred_greeks[i][1])
    pred_theta.append(pred_greeks[i][2])
    pred_vega.append(pred_greeks[i][3])
    pred_rho.append(pred_greeks[i][4])
    
    
# output to table for review
output_table = open('real_world_val/output_table.csv', "w")
output_table.write("strike,underlying,maturity,volatility,interestrate,call_price,pred_price,delta,pred_delta,theta,pred_theta,vega,pred_vega,rho,pred_rho\n")
for i in range(val_sample_size):
    output_table.write(str(val_x_vals[i][0].item()) + "," + str(val_x_vals[i][1].item()) + "," + str(val_x_vals[i][2].item()) + "," + str(val_x_vals[i][3].item()) + "," + str(val_x_vals[i][4].item()) + "," +
                       str(val_prices[i]) +"," + str(pred_price[i])+"," +
                       str(delta[i].item()) +"," + str(pred_delta[i].item()) +"," +
                       str(theta[i].item()) +"," + str(pred_theta[i].item()) + "," +
                       str(vega[i].item()) +"," + str(pred_vega[i].item()) +"," +
                       str(rho[i].item()) +"," + str(pred_rho[i].item()) + "\n")
output_table.close()

test_error = rmse_metric(val_prices, pred_price)
print("test error: " + str(test_error))
delta_error = (rmse_metric(delta, pred_delta))
theta_error = (rmse_metric(theta, pred_theta))
vega_errors = (rmse_metric(vega, pred_vega))
rho_errors = (rmse_metric(rho, pred_rho))

plt.scatter(val_prices, pred_price, label='pred price vs actual')
plt.title('price MSE: ' + str(test_error))
plt.xlabel('actual price')
plt.ylabel('pred price')
plt.savefig('real_world_val/test_price' + '.png', bbox_inches="tight")
plt.clf()

# # ADD MSE'S to this plot
plt.scatter(delta, pred_delta, label='delta test error')
plt.title('delta MSE: ' + str(delta_error))
plt.xlabel('actual delta')
plt.ylabel('pred delta')
plt.savefig('real_world_val/test_delta' + '.png', bbox_inches="tight")
plt.clf()
plt.scatter(theta, pred_theta, label='theta test error')
plt.title('theta MSE: ' + str(theta_error))
plt.xlabel('actual theta')
plt.ylabel('pred theta')
plt.savefig('real_world_val/test_theta' + '.png', bbox_inches="tight")
plt.clf()
plt.scatter(vega, pred_vega, label='vega test error')
plt.title('vega MSE: ' + str(vega_errors))
plt.xlabel('actual vega')
plt.ylabel('pred vega')
plt.savefig('real_world_val/test_vega' + '.png', bbox_inches="tight")
plt.clf()
plt.scatter(rho, pred_rho, label='rho test error')
plt.xlabel('actual rho')
plt.ylabel('pred rho')
plt.title('rho MSE: ' + str(rho_errors))
plt.savefig('real_world_val/test_rho' + '.png', bbox_inches="tight")
plt.clf()