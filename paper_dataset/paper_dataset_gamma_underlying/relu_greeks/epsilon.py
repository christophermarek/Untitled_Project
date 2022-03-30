from math import sqrt
from re import A
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal
import random
from torch import nn, optim
import pandas as pd
from scipy.stats import gamma, beta

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
    torch.nn.Linear(6, 400),
    torch.nn.ReLU(),
    torch.nn.Linear(400, 400),
    torch.nn.ReLU(),
    torch.nn.Linear(400, 400),
    torch.nn.ReLU(),
    torch.nn.Linear(400, 1),
)


lossfn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.0001)


# load dataset
df = pd.read_csv('../autodiff_dataset.csv')
X = df.drop(columns=['call_price', 'delta', 'theta', 'vega', 'rho', 'epsilon'])
Y = df.drop(columns=['strike', 'underlying', 'maturity', 'volatility', 'interestrate', 'dividends', 'call_price','delta','theta','vega','rho'])

x_vals = torch.tensor(X.values, dtype=torch.float32)
prices = torch.tensor(Y.values, dtype=torch.float32)

num_epoches = 2000000

print('starting training')
# 100% of the array for training data, validation set is generated when we get there
for m in range(num_epoches):
    opt.zero_grad()
    # WHAT IF I PASS THE WHOLE SET INSTEAD OF JUST ONE IDICE AT A TIME?
    model_out = model(x_vals[m].float())
    train_y = torch.tensor([prices[m]])
    loss = lossfn(model_out, train_y)

    loss.backward()
    opt.step()

        
    # save models at checkpoints
    if m == 10000 or m == 100000 or m == 500000 or m == 1000000 or m == 1500000:
        torch.save(model.state_dict(), 'epsilon' + str(m) + '.ckpt')
        
torch.save(model.state_dict(), 'epsilon.ckpt')

        
print('done training model')






