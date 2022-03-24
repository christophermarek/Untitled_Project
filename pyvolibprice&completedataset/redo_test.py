from math import sqrt
import math
from re import A
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal
import random
from torch import nn, optim
import pandas as pd
from scipy.stats import gamma, beta
from py_vollib.black_scholes_merton import black_scholes_merton
from scipy import stats

# ensure reproducability
np.random.seed(0)
torch.manual_seed(0)

# helper to calculate errors for greek
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


model = torch.nn.Sequential(
    torch.nn.Linear(6, 10),
    torch.nn.SiLU(),
    torch.nn.Linear(10, 10),
    torch.nn.SiLU(),
    torch.nn.Linear(10, 10),
    torch.nn.SiLU(),
    torch.nn.Linear(10, 1),
    torch.nn.SiLU()
)


lossfn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.0001)

# old pricing function, no dividends
std_norm_cdf = Normal(0, 1).cdf
# def std_norm_pdf(x): return torch.exp(Normal(0, 1).log_prob(x))
# def bs_price(right, K, S, T, sigma, r):
#     d_1 = (1 / (sigma * torch.sqrt(T))) * \
#           (torch.log(S / K) + (r + (torch.square(sigma) / 2)) * T)
#     d_2 = d_1 - sigma * torch.sqrt(T)

#     if right == "C":
#         C = std_norm_cdf(d_1) * S - std_norm_cdf(d_2) * \
#         K * torch.exp(-r * T)
#         return C

#     elif right == "P":
#         P = std_norm_cdf(-d_2) * K * torch.exp(-r * T) - \
#             std_norm_cdf(-d_1) * S
#         return P


def black_scholes (cp, s, k, t, v, rf, div):
        """ Price an option using the Black-Scholes model.
        cp: +1/-1 for call/put
        s: initial stock price
        k: strike price
        t: expiration time
        v: volatility
        rf: risk-free rate
        div: dividend
        """

        d1 = (torch.log(s/k)+(rf-div+0.5*torch.pow(v,2))*t)/(v*torch.sqrt(t))
        d2 = d1 - v*torch.sqrt(t)

        optprice = (cp*s*torch.exp(-div*t)*std_norm_cdf(cp*d1)) - (cp*k*torch.exp(-rf*t)*std_norm_cdf(cp*d2))
        return optprice

def gen_dataset(size):

    # will reset file every time it runs
    file = open('autodiff_dataset.csv', "w")
    file.write(
        "strike,underlying,maturity,volatility,interestrate,dividends,call_price,delta,theta,vega,rho,epsilon\n")

    for i in range(size):
        right = "C"
        
        # https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=2513&context=gradreports
        # how the dataset was calculated 
        
        # Strike price
        K = torch.tensor(gamma.rvs(100, size=1, scale=1)[0], requires_grad=True)
        # Underlying price Delta: {S.grad}
        S = torch.tensor(random.uniform(50, 200), requires_grad=True)
        # total time to expiry left in years Theta: {T.grad}
        T = torch.tensor(random.uniform(0.014, 1), requires_grad=True)
        # volatility "Vega: {sigma.grad}
        # sigma = torch.tensor(random.uniform(0.1, 0.4), requires_grad=True)
        sigma = torch.tensor(beta.rvs(a=2, b=5, size=1)[0] + 0.001, requires_grad=True)
        # risk free interest rate Rho: {r.grad}
        r = torch.tensor(random.uniform(0.01, 0.18), requires_grad=True)
        # dividend rate
        div_rate = torch.tensor(random.uniform(0.00, 0.18), requires_grad=True)

        # cp: +1/-1 for call/put
        # s: initial stock price
        # k: strike price
        # t: expiration time
        # v: volatility
        # rf: risk-free rate
        # div: dividend
        price = black_scholes(1, S, K, T, sigma, r, div_rate)

        # price = bs_price(right, K, S, T, sigma, r)

        price.backward()

        file.write(str(K.item()) + "," + str(S.item()) +
                   "," + str(T.item()) + "," + str(sigma.item()) + "," + str(r.item()) + "," + str(div_rate.item()) + "," + str(price.item()) + "," + str(S.grad.item()) +
                   "," + str(T.grad.item()) + "," + str(sigma.grad.item()) + "," + str(r.grad.item()) + "," + str(div_rate.grad.item()) + "\n")
    print("done generating data")

    file.close()


gen_dataset(1000000)

# load dataset
df = pd.read_csv('autodiff_dataset.csv')
X = df.drop(columns=['call_price', 'delta', 'theta', 'vega', 'rho', 'epsilon'])
Y = df.drop(columns=['strike', 'underlying', 'maturity', 'volatility', 'interestrate', 'dividends', 'delta','theta','vega','rho', 'epsilon'])
GREEKS = df.drop(columns=['strike', 'underlying', 'maturity', 'volatility', 'interestrate', 'dividends', 'call_price'])

x_vals = torch.tensor(X.values, dtype=torch.float32)
prices = torch.tensor(Y.values, dtype=torch.float32)
# can this just be an array instead, dont need to track grad as this is only used for validation
greeks = np.array(GREEKS.values)

num_epoches = 1000000

training_avg = 0
count = 0
training_avgs = []
training_loss = []


test_error = []
delta_error = []
theta_error = []
vega_error = []
rho_error = []
        
# 100% of the array for training data, validation set is generated when we get there
for m in range(num_epoches):
    opt.zero_grad()
    # WHAT IF I PASS THE WHOLE SET INSTEAD OF JUST ONE IDICE AT A TIME?
    model_out = model(x_vals[m].float())
    train_y = torch.tensor([prices[m]])
    loss = lossfn(model_out, train_y)
    loss_item = loss.item()

    training_avg = training_avg + (loss_item - training_avg) / (count + 1)
    count += 1
    training_avgs.append(training_avg)
    training_loss.append(loss_item)

    loss.backward()
    opt.step()

    with open("output.txt", "a+") as file:
        file.write("epoch num + " + str(m) +
                   " with loss: " + str(loss_item) + "\n")
        file.close()
        
    # save models at checkpoints
    if m == 10000 or m == 100000 or m == 500000 or m == 1000000 or m == 1500000:
        torch.save(model.state_dict(), 'auto_diff_greeks+' + str(m) + '.ckpt')
        

    # Now randomly generate sample data every 100 iters to run a validation during training to evaluate loss change over time
    if m % 100 == 0:
        # validation set of 1000 samples, probably dont need more than this? Good question to ask in office hours though
        test_x_vals = []
        test_prices = []
        test_greeks = []
        # generate validation set
        for i in range(100):
            # https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=2513&context=gradreports
            # how the dataset was calculated 
            
            # Strike price
            K = torch.tensor(gamma.rvs(100, size=1, scale=1)[0], requires_grad=True)
            # Underlying price Delta: {S.grad}
            S = torch.tensor(random.uniform(50, 200), requires_grad=True)
            # total time to expiry left in years Theta: {T.grad}
            T = torch.tensor(random.uniform(0.014, 1), requires_grad=True)
            # volatility "Vega: {sigma.grad}
            # sigma = torch.tensor(random.uniform(0.1, 0.4), requires_grad=True)
            sigma = torch.tensor(beta.rvs(a=2, b=5, size=1)[0] + 0.001, requires_grad=True)
            # risk free interest rate Rho: {r.grad}
            r = torch.tensor(random.uniform(0.01, 0.18), requires_grad=True)
            # dividend rate
            div_rate = torch.tensor(random.uniform(0.00, 0.18), requires_grad=True)

            # cp: +1/-1 for call/put
            # s: initial stock price
            # k: strike price
            # t: expiration time
            # v: volatility
            # rf: risk-free rate
            # div: dividend
            price = black_scholes(1, S, K, T, sigma, r, div_rate)

            price.backward()
            
            test_x_vals.append(torch.tensor(
            [K.item(), S.item(), T.item(), sigma.item(), r.item(), div_rate.item()], dtype=torch.float32))
            test_prices.append(price.item())
            
            # "strike,underlying,maturity,volatility,interestrate,call_price,delta,theta,vega,rho\n")
            test_greeks.append(np.array([S.grad.item(), T.grad.item(), sigma.grad.item(), r.grad.item(), div_rate.grad.item()]))
        
        
        pred_prices = []
        pred_greeks = []
        # run model on validation set
        for i in range(len(test_x_vals)):
            test_x_vals[i].requires_grad = True
            z = model(test_x_vals[i])
            pred_prices.append(z.item())
            z.sum().backward()
            pred_greeks.append(test_x_vals[i].grad)
            
        test_error.append(rmse_metric(test_prices, pred_prices))
        # THIS IS AN EASY INDEX ERROR TO MIXUP SINCE THE GREEKS AND XVALUES uSE DIFFERENT INDICES
        
        test_delta, test_theta, test_vega, test_rho, pred_delta, pred_theta, pred_vega, pred_rho = [], [], [], [], [], [], [], []
        for i in range(len(pred_greeks)):
            pred_delta.append(pred_greeks[i][1])
            pred_theta.append(pred_greeks[i][2])
            pred_vega.append(pred_greeks[i][3])
            pred_rho.append(pred_greeks[i][4])
        for i in range(len(test_greeks)):
            test_delta.append(test_greeks[i][0])
            test_theta.append(test_greeks[i][1])
            test_vega.append(test_greeks[i][2])
            test_rho.append(test_greeks[i][3])
            
        
        delta_error.append(rmse_metric(test_delta, pred_delta))
        theta_error.append(rmse_metric(test_theta, pred_theta))
        vega_error.append(rmse_metric(test_vega, pred_vega))
        rho_error.append(rmse_metric(test_rho, pred_rho))

print("training complete")
print('generating plots of training error and of the test errors')

plt.plot(list(range(0, len(test_error))), test_error,  label='price test errors')
plt.plot(list(range(0, len(test_error))), delta_error,  label='delta test errors')
plt.plot(list(range(0, len(test_error))), theta_error,  label='theta test errors')
plt.plot(list(range(0, len(test_error))), vega_error,  label='vega test errors')
plt.plot(list(range(0, len(test_error))), rho_error,  label='rho test errors')
plt.title('test errors')
plt.legend()
plt.savefig('price_test_error' + '.png', bbox_inches="tight")
plt.clf()

plt.plot(list(range(0, count)), training_avgs,  label='price training average error over time')
plt.title('average training errors')
plt.legend()
plt.savefig("avgtrain" + '.png', bbox_inches="tight")
plt.clf()

plt.plot(list(range(0, count)), training_loss,  label='price training error over time')
plt.title('exact training errors')
plt.legend()
plt.savefig("stricttrain" + '.png', bbox_inches="tight")
plt.clf()

# save model for easy testing of final output
torch.save(model.state_dict(), 'done+' + str(num_epoches) + '.ckpt')
# model.load_state_dict(torch.load('auto_diff_greeks+' + str(num_epoches) + '.ckpt'))
# model.load_state_dict(torch.load('silu1mil.ckpt'))


print('now evaluating validation set')
# LAST PART IS TO DO THE VALIDATION SET TO OUTPUT THE RESIDUAL CHARTS

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


#  we randomly generate the validation set after, since it is all random data anyways.
val_x_vals = []
val_prices = []
val_greeks = []
# generate validation set
# 20% size of test set
print('generating validation set')
for i in range(int(num_epoches* 0.2)):
    # https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=2513&context=gradreports
    # how the dataset was calculated 
        
    # Strike price
    K = torch.tensor(gamma.rvs(100, size=1, scale=1)[0], requires_grad=True)
    # Underlying price Delta: {S.grad}
    S = torch.tensor(random.uniform(50, 200), requires_grad=True)
    # total time to expiry left in years Theta: {T.grad}
    T = torch.tensor(random.uniform(0.014, 1), requires_grad=True)
    # volatility "Vega: {sigma.grad}
    # sigma = torch.tensor(random.uniform(0.1, 0.4), requires_grad=True)
    sigma = torch.tensor(beta.rvs(a=2, b=5, size=1)[0] + 0.001, requires_grad=True)
    # risk free interest rate Rho: {r.grad}
    r = torch.tensor(random.uniform(0.01, 0.18), requires_grad=True)
    # dividend rate
    div_rate = torch.tensor(random.uniform(0.00, 0.18), requires_grad=True)

    # cp: +1/-1 for call/put
    # s: initial stock price
    # k: strike price
    # t: expiration time
    # v: volatility
    # rf: risk-free rate
    # div: dividend
    price = black_scholes(1, S, K, T, sigma, r, div_rate)

    price.backward()
            
    val_x_vals.append(torch.tensor(
    [K.item(), S.item(), T.item(), sigma.item(), r.item(), div_rate.item()], dtype=torch.float32))
    val_prices.append(price.item())
            
    # "strike,underlying,maturity,volatility,interestrate,call_price,delta,theta,vega,rho\n")
    val_greeks.append(np.array([S.grad.item(), T.grad.item(), sigma.grad.item(), r.grad.item(), div_rate.grad.item()]))


# use last 20% as a validation set
for i in range(int(num_epoches*0.2)):
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
output_table = open('output_table.csv', "w")
output_table.write("strike,underlying,maturity,volatility,interestrate,call_price,pred_price,delta,pred_delta,theta,pred_theta,vega,pred_vega,rho,pred_rho\n")
for i in range(10):
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
plt.savefig('final_test_price' + '.png', bbox_inches="tight")
plt.clf()

# # ADD MSE'S to this plot
plt.scatter(delta, pred_delta, label='delta test error')
plt.title('delta MSE: ' + str(delta_error))
plt.xlabel('actual delta')
plt.ylabel('pred delta')
plt.savefig('final_test_delta' + '.png', bbox_inches="tight")
plt.clf()
plt.scatter(theta, pred_theta, label='theta test error')
plt.title('theta MSE: ' + str(theta_error))
plt.xlabel('actual theta')
plt.ylabel('pred theta')
plt.savefig('final_test_theta' + '.png', bbox_inches="tight")
plt.clf()
plt.scatter(vega, pred_vega, label='vega test error')
plt.title('vega MSE: ' + str(vega_errors))
plt.xlabel('actual vega')
plt.ylabel('pred vega')
plt.savefig('final_test_vega' + '.png', bbox_inches="tight")
plt.clf()
plt.scatter(rho, pred_rho, label='rho test error')
plt.xlabel('actual rho')
plt.ylabel('pred rho')
plt.title('rho MSE: ' + str(rho_errors))
plt.savefig('final_test_rho' + '.png', bbox_inches="tight")
plt.clf()


# RESIDUAL PLOT WOULD BE ACTUAL On x axis and residuals on y axis
res_prices = []
res_delta = []
res_theta = []
res_vega = []
res_rho = []

for i in range(len(val_prices)):
    res_prices.append(val_prices[i] - pred_price[i])
    res_delta.append(delta[i] - pred_delta[i])
    res_theta.append(theta[i] - pred_theta[i])
    res_vega.append(vega[i] - pred_vega[i])
    res_rho.append(rho[i] - pred_rho[i])

    
plt.scatter(val_prices, res_prices, label='delta test error')
plt.title('Residual plot - price MSE: ' + str(test_error))
plt.xlabel('actual prices')
plt.ylabel('actual - predicted price')
plt.savefig('price_residual_plot' + '.png', bbox_inches="tight")
plt.clf()

    
plt.scatter(delta, res_delta, label='delta test error')
plt.title('Residual plot - delta MSE: ' + str(delta_error))
plt.xlabel('actual delta')
plt.ylabel('actual - predicted delta')
plt.savefig('delta_residual_plot' + '.png', bbox_inches="tight")
plt.clf()

plt.scatter(theta, res_theta, label='theta test error')
plt.title('Residual plot - theta MSE: ' + str(theta_error))
plt.xlabel('actual theta')
plt.ylabel('actual - predicted theta')
plt.savefig('theta_residual_plot' + '.png', bbox_inches="tight")
plt.clf()

plt.scatter(vega, res_vega, label='vega test error')
plt.title('Residual plot - vega MSE: ' + str(theta_error))
plt.xlabel('actual vega')
plt.ylabel('actual - predicted vega')
plt.savefig('vega_residual_plot' + '.png', bbox_inches="tight")
plt.clf()

plt.scatter(rho, res_rho, label='rho test error')
plt.title('rho MSE: ' + str(theta_error))
plt.xlabel('actual rho')
plt.ylabel('actual - predicted rho')
plt.savefig('rho_residual_plot' + '.png', bbox_inches="tight")
plt.clf()



