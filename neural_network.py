# Definetly dont use all these imports anymore
from array import array
from math import nan
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# The "forward pass" refers to calculation process, values of the output layers from the inputs data. It's traversing through all neurons from first to last layer.
# A loss function is calculated from the output values.
# And then "backward pass" refers to process of counting changes in weights (de facto learning), using gradient descent algorithm (or similar). Computation is made from last layer, backward to the first layer.

# Backward and forward pass makes together one "iteration".

# During one iteration, you usually pass a subset of the data set, which is called "mini-batch" or "batch"
# (however, "batch" can also mean an entire set, hence the prefix "mini")

# "Epoch" means passing the entire data set in batches.
# One epoch contains (number_of_items / batch_size) iterations


# https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=2513&context=gradreports
# important i read this, keep in project doc

np.random.seed(0)
torch.manual_seed(0)

# load data into train set
# where output_columns are the columns we are keeping for train/test
def loadData(fileDir, output_columns):
    fileDir = 'generated_datasets/' + fileDir + '.csv'
        
    print(output_columns)

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

    # Sort by time to maturity so model has sense of time NOTE: important for meaningful predictions
    df = df.sort_values(by=['timetomaturity'], ascending=True)
    # print('PRINTING FIRST 5 ROWS OF DATASET')
    # print(df.head())

    # price not used for greeks, but it is actually.
    # df = df.drop(columns="BS-Call")

    # X = df.drop(columns=['delta', 'gamma', 'rho', 'theta', 'vega'])
    # X = df.drop(columns="BS-Call")
    # y = df[['delta', 'gamma', 'rho', 'theta', 'vega']]
    # X = df.drop(columns=output_columns)
    # y = df[output_columns]
    # y = df["Bs-Call"]
    X = df.drop(columns=['BS-Call', 'delta', 'gamma', 'rho', 'theta', 'vega'])
    print('printing x head')
    print(X.head())
    print('printing y head')
    y = df['BS-Call']
    print(y.head())
    print('done printing y head')
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
    print('dataset loaded')
    return [X_train, X_test, y_train, y_test]


def trainModel(model, optimizer, lossFN, input, output, numEpochs):
    
    model.train()
    # begin to train
    logger = list()
    logger.append(model.name)
    flag = False
    for i in range(numEpochs):
        if flag:
            break
        print('STEP: ', i)
        logger.append('STEP: ' + str(i))
        def closure():
            optimizer.zero_grad()
            out = model(input.float())
            loss = lossFN(out, output.float())
            # print('loss:', loss.item())
            logger.append('loss: ' + str(loss.item()))
            loss.backward()
            return loss
        optimizer.step(closure)
        
        if 'loss: nan' in logger:
            print('nan loss detected, ending training \n')
            logger.append('NAN LOSS DETECTED')
            break
        
    logger.append('DONE')
    
    with open('modeltrainingoutput.txt', 'a+') as f:
        for log_entry in logger:
            f.write(log_entry + '\n')


def testModel(model, lossFN, testInput, testOutput):
    # begin to predict, no need to track gradient here
    model.eval()
    with torch.no_grad():
        pred = model(testInput.float())
    cost = lossFN(pred, testOutput.float())
    print("mean squared error:", cost.item())
    return [pred, cost.item()]
