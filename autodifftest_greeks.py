from math import sqrt
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal
import random
from torch import nn, optim


model = torch.nn.Sequential(
    torch.nn.Linear(5, 3),
    torch.nn.Sigmoid(),
    torch.nn.Linear(3, 3),
    torch.nn.Sigmoid(),
    torch.nn.Linear(3, 1)
)
# model_3_droupout = torch.nn.Sequential(
#     torch.nn.Linear(5, 3),
#     nn.Dropout(0.25),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(3, 3),
#     nn.Dropout(0.25),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(3, 1)
# )

# model_4 = torch.nn.Sequential(
#     torch.nn.Linear(5, 4),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(4, 4),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(4, 1)
# )
# model_4_droupout = torch.nn.Sequential(
#     torch.nn.Linear(5, 4),
#     nn.Dropout(0.25),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(4, 4),
#     nn.Dropout(0.25),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(4, 1)
# )

# model_3_2layers = torch.nn.Sequential(
#     torch.nn.Linear(5, 3),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(3, 3),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(3, 3),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(3, 1)
# )
# model_4_2layers = torch.nn.Sequential(
#     torch.nn.Linear(5, 4),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(4, 4),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(4, 4),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(4, 1)
# )

# model_10_1layers = torch.nn.Sequential(
#     torch.nn.Linear(5, 10),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(10, 10),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(10, 1)
# )

# model_10_2layers = torch.nn.Sequential(
#     torch.nn.Linear(5, 10),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(10, 10),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(10, 10),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(10, 1)
# )


lossfn = nn.MSELoss()

# def model_run(lr_param, model, plotname):
    
opt = torch.optim.Adam(model.parameters(), lr=0.0001)

def rmse_metric(actual, predicted):
        sum_error = 0.0
        for i in range(len(actual)):
            prediction_error = predicted[i] - actual[i]
            sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(actual))
        return sqrt(mean_error)
    
# GENERATE DATASET
std_norm_cdf = Normal(0, 1).cdf
def std_norm_pdf(x): return torch.exp(Normal(0, 1).log_prob(x))


def bs_price(right, K, S, T, sigma, r):
    d_1 = (1 / (sigma * torch.sqrt(T))) * \
        (torch.log(S / K) + (r + (torch.square(sigma) / 2)) * T)
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

num_iterations = 100

num_epoches = 100

training_errors = []
test_errors = []

delta_errors = []
theta_errors = []
theta_errors = []
vega_errors = []
vega_errors = []
rho_errors = []


for n in range(num_iterations):
    
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
        x_vals.append(torch.tensor(
            [K.item(), S.item(), T.item(), sigma.item(), r.item()], dtype=torch.float32))
        prices.append(price.item())
        greeks.append(np.array([K.item(), S.grad.item(),
                    sigma.grad.item(), T.grad.item(), r.grad.item()]))
    
    
    print(x_vals[0])
    
    training_avg = 0
    count = 0
    # 80% of the array for training data
    for m in range(int((num_epoches*0.8))):
        opt.zero_grad()
        model_out = model(x_vals)
        train_y = torch.tensor([prices])
        loss = lossfn(model_out, train_y)
        loss_item = loss.item()
            
        training_avg = training_avg + (loss_item - training_avg) / (count + 1)
        count += 1
            
        loss.backward()
        opt.step()

        # with open("output.txt", "a+") as file:
        #     file.write("iter_num: " +  str(n) + " epoch num + " + str(m) + " with loss: " + str(loss_item) + "\n")
        #     file.close()


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

    # use last 20% as a validation set
    for i in range(int(num_epoches*0.8), num_epoches):
            
        x_vals[i].requires_grad = True
        z = model(x_vals[i])
        pred_price.append(z.item())
        z.sum().backward()

        pred_greeks.append(x_vals[i].grad)

        delta.append(greeks[i][1])
        theta.append(greeks[i][2])
        vega.append(greeks[i][3])
        rho.append(greeks[i][4])

    for i in range(len(pred_greeks)):
        pred_delta.append(pred_greeks[i][1])
        pred_theta.append(pred_greeks[i][2])
        pred_vega.append(pred_greeks[i][3])
        pred_rho.append(pred_greeks[i][4])
        
    test_error = rmse_metric(prices[-int(num_epoches*0.2):], pred_price)
    print(test_error)
        
        # plt.clf()
        # plt.scatter(prices[-int(num_epoches*0.2):], pred_price, marker=".", label="price residual")
        # plt.xlabel('Test call price')
        # plt.ylabel('Pred call price')
        # plt.title('MSE: ' + str(rmse_metric(prices[-int(num_epoches*0.2):], pred_price)))
        # plt.savefig('test/Call2'+ str(n) + '.png', bbox_inches="tight")
        # plt.clf()
        # plt.scatter(delta, pred_delta, marker=".", label="delta residual")
        # plt.xlabel('Test delta')
        # plt.ylabel('Pred delta')
        # plt.title('MSE: ' + str(rmse_metric(delta, pred_delta)))
        # plt.savefig('test/delta2'+ str(n) + '.png', bbox_inches="tight")
        # plt.clf()
        # plt.scatter(theta, pred_theta, marker=".", label="theta residual")
        # plt.xlabel('Test theta')
        # plt.ylabel('Pred theta')
        # plt.title('MSE: ' + str(rmse_metric(theta, pred_theta)))
        # plt.savefig('test/theta2'+ str(n) + '.png', bbox_inches="tight")
        # plt.clf()
        # plt.scatter(vega, pred_vega, marker=".", label="vega residual")
        # plt.xlabel('Test vega')
        # plt.ylabel('Pred vega')
        # plt.title('MSE: ' + str(rmse_metric(vega, pred_vega)))
        # plt.savefig('test/vega2'+ str(n) + '.png', bbox_inches="tight")
        # plt.clf()
        # plt.scatter(rho, pred_rho, marker=".", label="rho residual")
        # plt.xlabel('Test rho')
        # plt.ylabel('Pred rho')
        # plt.title('MSE: ' + str(rmse_metric(rho, pred_rho)))
        # plt.savefig('test/rho2'+ str(n) + '.png', bbox_inches="tight")
        # plt.clf()
        
    training_errors.append(training_avg)
    test_errors.append(test_error)
    delta_errors.append(rmse_metric(delta, pred_delta))
    theta_errors.append(rmse_metric(theta, pred_theta))
    vega_errors.append(rmse_metric(vega, pred_vega))
    rho_errors.append(rmse_metric(rho, pred_rho))
        
        

# torch.save(model.state_dict(), 'models/autodifftest.ckpt')
# print(training_errors)
# print(test_errors)
# plt.plot(list(range(0, num_iterations)), training_errors,  label='price training error')
# plt.plot(list(range(0, num_iterations)), test_errors, label='price test error')
# plt.title('training and test errors')
# plt.legend()
# plt.savefig('test/justprices'+ "fix" + '.png', bbox_inches="tight")
# # plt.show()
# plt.clf()
# plt.plot(list(range(0, num_iterations)), training_errors,  label='price training error')
# plt.plot(list(range(0, num_iterations)), test_errors, label='price test error')
# plt.plot(list(range(0, num_iterations)), delta_errors, label='delta test error')
# plt.plot(list(range(0, num_iterations)), theta_errors, label='theta test error')
# plt.plot(list(range(0, num_iterations)), vega_errors, label='vega test error')
# plt.plot(list(range(0, num_iterations)), rho_errors, label='rho test error')
# plt.title('training and test errors plus greeks')
# plt.legend()
# plt.savefig('test/'+ "fix" + '.png', bbox_inches="tight")
# # plt.show()
# plt.clf()



# def main():
#     lrs = [0.1, 0.01, 0.001, 0.0001]
#     # test model with dropout layers
#     # test model with 4 neurons
#     # test with two hidden layers (3,3), (4,4)
#     # test with more hidden layer neurons
    
#     # Then this should be enough plots, and it shouldnt take too long if we arent doing a ton of iters

#     for lr in lrs:
#         print(lr)
#         model_run(lr, model_3, 'model_3'+str(lr))
#         model_run(lr, model_3_droupout, 'model_3_droupout'+str(lr))
#         model_run(lr, model_4, 'model_4'+str(lr))
#         model_run(lr, model_4_droupout, 'model_4_droupout'+str(lr))
#         model_run(lr, model_3_2layers, 'model_3_2layers'+str(lr))
#         model_run(lr, model_4_2layers, 'model_4_2layers'+str(lr))
#         model_run(lr, model_10_1layers, 'model_10_1layers'+str(lr))
#         model_run(lr, model_10_2layers, 'model_10_2layers'+str(lr))
       
# main()

