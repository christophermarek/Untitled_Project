import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torch import nn, optim
import pandas as pd
from sklearn.model_selection import train_test_split

# https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=2513&context=gradreports


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
                          nn.LogSoftmax(dim=1))
    print(model)

    # https://stackoverflow.com/questions/46170814/how-to-train-pytorch-model-with-numpy-data-and-batch-size
    # THis guy has a good way of setting up his code, and the guy answering explains how to do it right.
    # he even links the docs, cause the data needs to be formatted a specific way for pytorhc

    # https://github.com/pytorch/examples/blob/master/mnist/main.py#L37
    # THis official example I can just copy most of it, it looks right too and can load local files
    # like the other tutorial did instead of a slow to load df.
# neural_network()


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # convert to tensors
    X_train = torch.tensor(X_train.values)
    y_train = torch.tensor(y_train.values)
    X_test = torch.tensor(X_test.values)
    y_test = torch.tensor(y_test.values)

    # build the model
    # seq = Sequence()
    # seq.double()
    # criterion = nn.MSELoss()
    # # use LBFGS as optimizer since we can load the whole data to train
    # optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    # #begin to train
    # for i in range(opt.steps):
    #     print('STEP: ', i)
    #     def closure():
    #         optimizer.zero_grad()
    #         out = seq(input)
    #         loss = criterion(out, target)
    #         print('loss:', loss.item())
    #         loss.backward()
    #         return loss
    #     optimizer.step(closure)
    #     # begin to predict, no need to track gradient here
    #     with torch.no_grad():
    #         future = 1000
    #         pred = seq(test_input, future=future)
    #         loss = criterion(pred[:, :-future], test_target)
    #         print('test loss:', loss.item())
    #         y = pred.detach().numpy()
    #     # draw the result
    #     plt.figure(figsize=(30,10))
    #     plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    #     plt.xlabel('x', fontsize=20)
    #     plt.ylabel('y', fontsize=20)
    #     plt.xticks(fontsize=20)
    #     plt.yticks(fontsize=20)
    #     def draw(yi, color):
    #         plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
    #         plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
    #     draw(y[0], 'r')
    #     draw(y[1], 'g')
    #     draw(y[2], 'b')
    #     plt.savefig('predict%d.pdf'%i)
    #     plt.close()


main()
