import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torch import nn, optim
import pandas as pd
from sklearn.model_selection import train_test_split

# https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=2513&context=gradreports


# 6 outputs one for each greek.
# But thats later once I set this up properly
def neural_network():

    # Get dataset
    filePath = 'output/demo_file.csv'

    # output table of data showing few rows
    df = False

    try:
        df = pd.read_csv(filePath)
        print(df.head())
    except:
        print('error opening file')

    X_train, X_test, y_train, y_test = train_test_split(df.drop('BS-Call', axis=1),
                                                        df['BS-Call'], test_size=0.2)

    # wonder if i should round output price column, we will see later

    # What type of neural network would i use
    # linear for now simple
    # The nn.Sequential wraps the layers in the network. There are three linear layers with ReLU activation ( a simple function which allows
    # positive values to pass through, whereas negative values are modified to zero )

    #      The input
    # layer has six nodes (one for each input parameter). The hidden layers have ten nodes each. The
    # output layer has one node (for the predicted price)
    input_size = 6
    # first hidden layer 128 neurons, seconq 64
    hidden_sizes = [10, 10]
    output_size = 1
    # input data passes through the model in sequential order, linear[hl1] -> relu -> linear[hl2] -> relu -> output -> softmax
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.LogSoftmax)
    print(model)

    # https://stackoverflow.com/questions/46170814/how-to-train-pytorch-model-with-numpy-data-and-batch-size
    # THis guy has a good way of setting up his code, and the guy answering explains how to do it right.
    # he even links the docs, cause the data needs to be formatted a specific way for pytorhc

    # https://github.com/pytorch/examples/blob/master/mnist/main.py#L37
    # THis official example I can just copy most of it, it looks right too and can load local files
    # like the other tutorial did instead of a slow to load df.
# neural_network()


class BlackScholesModel_Simple(nn.Module):

    def __init__(self):
        input_size = 5
        hidden_sizes = [10, 10]
        output_size = 1
        super(BlackScholesModel_Simple, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[1], output_size))

    def forward(self, input):
        x = self.model(input)
        return x


def main():
    # The "forward pass" refers to calculation process, values of the output layers from the inputs data. It's traversing through all neurons from first to last layer.
    # A loss function is calculated from the output values.
    # And then "backward pass" refers to process of counting changes in weights (de facto learning), using gradient descent algorithm (or similar). Computation is made from last layer, backward to the first layer.

    # Backward and forward pass makes together one "iteration".

    # During one iteration, you usually pass a subset of the data set, which is called "mini-batch" or "batch"
    # (however, "batch" can also mean an entire set, hence the prefix "mini")

    # "Epoch" means passing the entire data set in batches.
    # One epoch contains (number_of_items / batch_size) iterations

    print('from main')

    # set random seed to 0
    # why are we doing this
    np.random.seed(0)
    torch.manual_seed(0)

    # load data and make training set
    # Get dataset
    filePath = 'output/demo_file.csv'

    # output table of data showing few rows
    df = False

    try:
        df = pd.read_csv(filePath)
        # print(df.head())
    except:
        print('error opening file')

    X = df.drop(columns="BS-Call")
    y = df['BS-Call']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # convert to tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float64)
    X_test = torch.tensor(X_test.values, dtype=torch.float64)


    # Need to reshape y tensor so it has the same shape as X
    y_train = torch.tensor(y_train.values, dtype=torch.float64)
    y_test = torch.tensor(y_test.values, dtype=torch.float64)

    new_shape_ytrain = (len(y_train), 1)
    y_train = y_train.view(new_shape_ytrain)
    new_shape_ytest = (len(y_test), 1)
    y_test = y_test.view(new_shape_ytest)

    # build the model
    blackscholesmodel = BlackScholesModel_Simple()
    model = torch.load('model.ckpt')
    # model.eval()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    # lr = learning rate
    # Will need to test different learning rates and optimizers
    # optimizer = optim.LBFGS(blackscholesmodel.parameters(), lr=0.8)
    # #begin to train
    # for i in range(20):
    #     print('STEP: ', i)
    #     def closure():
    #         optimizer.zero_grad()
    #         out = blackscholesmodel(X_train.float())
    #         loss = criterion(out, y_train.float())
    #         # print('loss:', loss.item())
    #         loss.backward()
    #         return loss
    #     optimizer.step(closure)

    # After move my predictions to after the training
    # begin to predict, no need to track gradient here
    print(len(y_test.float()))
    with torch.no_grad():
        pred = blackscholesmodel(X_test.float())
        # loss = criterion(pred, y_test.float())
        # print('test loss:', loss.item())
    cost = criterion(pred, y_test.float())
    print("mean squared error:", cost.item())
    # Now probably plot them to see.
    # Maybe x axis is index
    # y axis is call value
    xAxis = []
    for n in range(100):
        xAxis.append(n)

    plt.scatter(xAxis, y_test.float()[0:100],marker=".", label="ytest", color='black')
    plt.scatter(xAxis, pred[0:100],marker=".", label="ml prediction", color='red')
    plt.legend()
    plt.show()

    # Save the model checkpoint
    torch.save(blackscholesmodel.state_dict(), 'model.ckpt')


main()
