from math import sqrt
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal
import random
from torch import nn, optim
import pandas as pd

# ensure reproducability
# np.random.seed(0)
# torch.manual_seed(0)

# helper to calculate errors for greek
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

# model = torch.nn.Sequential(
#     torch.nn.Linear(5, 3),
#     torch.nn.SiLU(),
#     torch.nn.Linear(3, 3),
#     torch.nn.SiLU(),
#     torch.nn.Linear(3, 3),
#     torch.nn.SiLU(),
#     torch.nn.Linear(3, 1),
#     torch.nn.SiLU(),
# )

lossfn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.0001)

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


def gen_dataset(size):

    # will reset file every time it runs
    file = open('autodiff_dataset.csv', "w")
    file.write(
        "strike,underlying,maturity,volatility,interestrate,call_price,delta,theta,vega,rho\n")

    for i in range(size):
        right = "C"
        moneyness = random.uniform(0.8, 1.2)
        underlyingPrice = 1.0
        strike = underlyingPrice / moneyness

        # Strike price
        K = torch.tensor(strike, requires_grad=True)
        # Underlying price Delta: {S.grad}
        S = torch.tensor(underlyingPrice, requires_grad=True)
        # total time to expiry left in years Theta: {T.grad}
        T = torch.tensor(random.uniform(0.014, 1), requires_grad=True)
        # volatility "Vega: {sigma.grad}
        sigma = torch.tensor(random.uniform(0.1, 0.4), requires_grad=True)
        # risk free interest rate Rho: {r.grad}
        r = torch.tensor(random.uniform(0.00, 0.1), requires_grad=True)

        price = bs_price(right, K, S, T, sigma, r)

        price.backward()

        file.write(str(K.item()) + "," + str(S.item()) +
                   "," + str(T.item()) + "," + str(sigma.item()) + "," + str(r.item()) + "," + str(price.item()) + "," + str(S.grad.item()) +
                   "," + str(T.grad.item()) + "," + str(sigma.grad.item()) + "," + str(r.grad.item()) + "\n")
    print("done generating data")

    file.close()


# gen_dataset(1000000)

# load dataset
df = pd.read_csv('autodiff_dataset.csv')
X = df.drop(columns=['call_price', 'delta', 'theta', 'vega', 'rho'])
Y = df.drop(columns=['strike', 'underlying', 'maturity', 'volatility', 'interestrate', 'call_price','theta','vega','rho'])
# GREEKS = df.drop(columns=['strike', 'underlying', 'maturity', 'volatility', 'interestrate', 'call_price'])

x_vals = torch.tensor(X.values, dtype=torch.float32)
prices = torch.tensor(Y.values, dtype=torch.float32)
# can this just be an array instead, dont need to track grad as this is only used for validation
# greeks = np.array(GREEKS.values)


num_epoches = 1000000

training_avg = 0
count = 0
training_avgs = []
training_loss = []


test_error = []
# delta_error = []
# theta_error = []
# vega_error = []
# rho_error = []
        
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
    if m == 10000 or m == 100000 or m == 500000 or m == 1000000:
        torch.save(model.state_dict(), 'auto_diff_greeks+' + str(m) + '.ckpt')
        

    # Now randomly generate sample data every 100 iters to run a validation during training to evaluate loss change over time
    if m % 100 == 0:
        # validation set of 1000 samples, probably dont need more than this? Good question to ask in office hours though
        test_x_vals = []
        test_prices = []
        test_greeks = []
        # generate validation set
        for i in range(100):
            right = "C"
            moneyness = random.uniform(0.8, 1.2)
            underlyingPrice = 1.0
            strike = underlyingPrice / moneyness

            # Strike price
            K = torch.tensor(strike, requires_grad=True)
            # Underlying price Delta: {S.grad}
            S = torch.tensor(underlyingPrice, requires_grad=True)
            # total time to expiry left in years Theta: {T.grad}
            T = torch.tensor(random.uniform(0.014, 1), requires_grad=True)
            # volatility "Vega: {sigma.grad}
            sigma = torch.tensor(random.uniform(0.1, 0.4), requires_grad=True)
            # risk free interest rate Rho: {r.grad}
            r = torch.tensor(random.uniform(0.00, 0.1), requires_grad=True)

            price = bs_price(right, K, S, T, sigma, r)

            price.backward()
            
            test_x_vals.append(torch.tensor(
            [K.item(), S.item(), T.item(), sigma.item(), r.item()], dtype=torch.float32))
            test_prices.append(S.grad.item())
            
        
        
        pred_prices = []
        for i in range(len(test_x_vals)):
            test_x_vals[i].requires_grad = True
            z = model(test_x_vals[i])
            pred_prices.append(z.item())
            z.sum().backward()
            # pred_greeks.append(test_x_vals[i].grad)
            
        test_error.append(rmse_metric(test_prices, pred_prices))


print("training complete")
print('generating plots of training error and of the test errors')

plt.plot(list(range(0, len(test_error))), test_error,  label='price test errors')
plt.title('test errors')
plt.legend()
plt.savefig('delta_output/price_test_error' + '.png', bbox_inches="tight")
plt.clf()

plt.plot(list(range(0, count)), training_avgs,  label='price training average error over time')
plt.title('average training errors')
plt.legend()
plt.savefig('delta_output/' + "avgtrain" + '.png', bbox_inches="tight")
plt.clf()

plt.plot(list(range(0, count)), training_loss,  label='price training error over time')
plt.title('exact training errors')
plt.legend()
plt.savefig('delta_output/' + "stricttrain" + '.png', bbox_inches="tight")
plt.clf()

# save model for easy testing of final output
torch.save(model.state_dict(), 'delta.ckpt')
# model.load_state_dict(torch.load('auto_diff_greeks+' + str(num_epoches) + '.ckpt'))
# model.load_state_dict(torch.load('auto_diff_greeks+10000.ckpt'))


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
    right = "C"
    moneyness = random.uniform(0.8, 1.2)
    underlyingPrice = 1.0
    strike = underlyingPrice / moneyness

    # Strike price
    K = torch.tensor(strike, requires_grad=True)
    # Underlying price Delta: {S.grad}
    S = torch.tensor(underlyingPrice, requires_grad=True)
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
    val_prices.append(S.grad.item())
            
    # "strike,underlying,maturity,volatility,interestrate,call_price,delta,theta,vega,rho\n")
    val_greeks.append(np.array([S.grad.item(), T.grad.item(), sigma.grad.item(), r.grad.item()]))


# use last 20% as a validation set
for i in range(int(num_epoches*0.2)):
    val_x_vals[i].requires_grad = True
    # print(x_vals[i])
    z = model(val_x_vals[i])
    pred_price.append(z.item())
    z.sum().backward()


test_error = rmse_metric(val_prices, pred_price)


plt.scatter(pred_price, val_prices, label='pred price vs actual')
plt.title('price MSE: ' + str(test_error))
plt.xlabel('pred delta')
plt.ylabel('actual delta')
plt.savefig('delta_output/final_test_price' + '.png', bbox_inches="tight")
plt.clf()
