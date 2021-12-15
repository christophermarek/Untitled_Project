import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torch import nn, optim
import pandas as pd
from sklearn.model_selection import train_test_split

# https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=2513&context=gradreports


# 6 outputs for 1 greek but thats later.

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
                                   nn.Linear(hidden_sizes[1], output_size),
                                   )

    def forward(self, input):
        x = self.model(input)
        return x

def trainModel(model, optimizer, lossFN, input, output, numEpochs):
    model.train()
    # begin to train
    for i in range(numEpochs):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = model(input.float())
            loss = lossFN(out, output.float())
            loss.backward()
            return loss
        optimizer.step(closure)

def testModel(model, lossFN, testInput, testOutput):
    # begin to predict, no need to track gradient here
    model.eval()
    with torch.no_grad():
        pred = model(testInput.float())
    cost = lossFN(pred, testOutput.float())
    print("mean squared error:", cost.item())
    return pred

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
    except:
        print('error opening file')

    # Sort by time to maturity so model has sense of time?
    # THis actually improves performance substantially, note this in description
    df = df.sort_values(by=['timetomaturity'], ascending=True)

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
    optimizer = optim.LBFGS(blackscholesmodel.parameters(), lr=0.5)

    # train model
    trainModel(blackscholesmodel, optimizer, criterion, X_train, y_train, 20)
    # test model and get output
    pred = testModel(blackscholesmodel, criterion, X_test, y_test)
    
    # NOW NEED A WAY TO QUANTIFY ACCURACY BETTER, I LIKE THE TWO PLOTS BUT THEY STILL ARE NOT GOOD COPY YET
    
    # Then results need to be saved somewhere so i can compare all models

    # Maybe x axis is index
    # y axis is call value
    xAxis = []
    for n in range(3000):
        xAxis.append(n)

    fig, ax = plt.subplots(2, 1)

    ax[0].scatter(xAxis, pred[0:3000],marker=".", label="ml prediction", color='red')
    ax[0].scatter(xAxis, y_test.float()[0:3000],marker=".", label="y_test", color='black')
    ax[0].legend()
    ax[1].scatter(y_test.float()[0:3000], pred[0:3000],marker=".")
    plt.show()

    # Save the model checkpoint
    torch.save(blackscholesmodel.state_dict(), 'model.ckpt')


main()
