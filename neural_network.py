import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
from torch import nn, optim
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import os

# https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=2513&context=gradreports
# 6 outputs for 1 greek but thats later.

def loadData(fileDir):

     # load data and make training set
    # Get dataset
    filePath = fileDir

    # output table of data showing few rows
    df = False
    try:
        df = pd.read_csv(filePath)
    except:
        print('error opening file')
        return False

    # Sort by time to maturity so model has sense of time?
    # THis actually improves performance substantially, note this in description
    # In description of execution, write a notes section and make this a point
    # - As seen in the literature review, keeping a time ordering is important for the ML
    # algoruthm to get a sense of time ordering.
    df = df.sort_values(by=['timetomaturity'], ascending=True)

    X = df.drop(columns="BS-Call")
    y = df['BS-Call']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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

    return [X_train, X_test, y_train, y_test]

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
                                   nn.ReLU(),
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

def main(trainOrTestMode, models, dataSet):
    # The "forward pass" refers to calculation process, values of the output layers from the inputs data. It's traversing through all neurons from first to last layer.
    # A loss function is calculated from the output values.
    # And then "backward pass" refers to process of counting changes in weights (de facto learning), using gradient descent algorithm (or similar). Computation is made from last layer, backward to the first layer.

    # Backward and forward pass makes together one "iteration".

    # During one iteration, you usually pass a subset of the data set, which is called "mini-batch" or "batch"
    # (however, "batch" can also mean an entire set, hence the prefix "mini")

    # "Epoch" means passing the entire data set in batches.
    # One epoch contains (number_of_items / batch_size) iterations

    # set random seed to 0
    # do this so results can be reproducable
    np.random.seed(0)
    torch.manual_seed(0)
    
    X_train, X_test, y_train, y_test = dataSet
        
    for model in models:
        
        # build the model
        runningModel = False
        if model == 'simpleblackscholes': runningModel = BlackScholesModel_Simple()
        if not model: 
            print('invalid model name passed')
            return

        if(trainOrTestMode == 1):
            runningModel.load_state_dict(torch.load(model + '.ckpt'))

        criterion = nn.MSELoss()

        # use LBFGS as optimizer since we can load the whole data to train
        # lr = learning rate
        # Will need to test different learning rates and optimizers
        optimizer = optim.LBFGS(runningModel.parameters(), lr=0.5)

        # train model
        if trainOrTestMode == 0 or trainOrTestMode == 2:
            trainModel(runningModel, optimizer, criterion, X_train, y_train, 20)
            torch.save(runningModel.state_dict(), model + '.ckpt')

        # test model and get output
        if trainOrTestMode == 1 or trainOrTestMode == 2:
            pred = testModel(runningModel, criterion, X_test, y_test)

            # Then results need to be saved somewhere so i can compare all models
            # NOW NEED A WAY TO QUANTIFY ACCURACY BETTER, I LIKE THE TWO PLOTS BUT THEY STILL ARE NOT GOOD COPY YET

            now = datetime.now()
            
            # Save plots to an output dir, title model name and date/time ran
            if not os.path.exists('model_output'):
                os.makedirs('model_output')
            if not os.path.exists('model_output/' + model):
                os.makedirs('model_output/' + model)
            if not os.path.exists('model_output/' + model + '/' + now.strftime("%d_%m_%Y_%H_%M_%S")):
                os.makedirs('model_output/' + model + '/' + now.strftime("%d_%m_%Y_%H_%M_%S"))
            
            # Maybe x axis is index
            # y axis is call value
            xAxis = []
            for n in range(300):
                xAxis.append(X_test[n][1])

            # fig, ax = plt.subplots(2, 1)

            # LABEL AXES FOR GOOD COPY
            # MAKE X VALUES TIME TO MATURITIES INSTEAD LOL THAT MAKES WAY MORE SENSE THAN JUST INDEX,
            # AND THEY ARE ORDERED BY TIME TO MATURITY ANYWAYS, EASIER TO SPOT A PATTERN.
            plt.scatter(xAxis, pred[0:300],marker=".", label="ml prediction", color='red')
            plt.scatter(xAxis, y_test.float()[0:300],marker=".", label="y_test", color='black')
            plt.xlabel('Time To Maturity')
            plt.ylabel('Option Price')
            plt.legend()
            plt.savefig('model_output/' + model + "/" + now.strftime("%d_%m_%Y_%H_%M_%S") + "/" + "vsMaturity" + '.png')
            plt.clf()

            
            plt.scatter(y_test.float()[0:300], pred[0:300],marker=".")
            plt.xlabel('Test Option Price')
            plt.ylabel('Predicted Option Price')
            plt.savefig('model_output/' + model + "/" +  now.strftime("%d_%m_%Y_%H_%M_%S") + "/" + "testvsprediction" + '.png')
            
def preRun():
    # get params
    mode = input("Enter 2 for train and test, 1 for test mode, 0 for train mode:  ")
    mode = int(mode)
    if not (mode in [0,1,2]):
        print('invalid mode paramaters')
        return

    models = [
        "simpleblackscholes"
    ]
    
    dataSetPath = 'output/demo_file.csv'
    dataSet = loadData(dataSetPath)
    if not dataSet: 
        print('invalid dataset path')
        return
    
    main(mode, models, dataSet)

preRun()
