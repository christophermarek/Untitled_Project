# Definetly dont use all these imports anymore
import sys
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
# It contains code too I didnt realize this before, shows how they generate their data exactly

# 6 outputs for 1 greek but thats later.

# ADD to readme explanation of each file and directory

# REMOVE TIMESTAMPS ON SAVED FILES IT JUST ADDS CLUTTER

# load data into train set
def loadData(fileDir):
    fileDir = 'generated_datasets/' + fileDir + '.csv'
    
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
    df = df.drop(columns="BS-Call")
    
    X = df.drop(columns=['iv','delta', 'gamma', 'rho', 'theta', 'vega'])
    y = df[['iv','delta', 'gamma', 'rho', 'theta', 'vega']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # convert to tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float64)
    X_test = torch.tensor(X_test.values, dtype=torch.float64)
    # Need to reshape y tensor so it has the same shape as X
    y_train = torch.tensor(y_train.values, dtype=torch.float64)
    y_test = torch.tensor(y_test.values, dtype=torch.float64)
    new_shape_ytrain = (len(y_train), 6)
    y_train = y_train.view(new_shape_ytrain)
    new_shape_ytest = (len(y_test), 6)
    y_test = y_test.view(new_shape_ytest)
    print('dataset loaded')
    return [X_train, X_test, y_train, y_test]



def trainModel(model, optimizer, lossFN, input, output, numEpochs):
    model.train()
    # begin to train
    for i in range(numEpochs):
        print('Epoch: ', i)
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
    return [pred, cost.item()]

def main(trainOrTestMode, models, dataSet, hyperparam_config):
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
    
    # Add note how relu makes models predict 0, especially with more layers. This is probably because the relu forces it to 0.
    
    # ADD a way to load multiple datasets
    
    X_train, X_test, y_train, y_test = dataSet
    print('in main neural_networks now')
    models_error = list()
    
    for model in models:
        for hyperParamConfigEntry in hyperparam_config:
            learningRate = hyperParamConfigEntry[0]
            numNeurons = hyperParamConfigEntry[1]

            # build the model
            runningModel = False
            # else:
            #     print(model)
            #     modelSplit = model.split(',')
            #     model_name = modelSplit[0]
            #     restOfModel = ''.join(modelSplit[1:])
            #     restOfModelSplit = restOfModel.split('.')
            #     numNeurons = int(restOfModelSplit[1][-2:])
            #     learningRate = restOfModelSplit[1][:-2]
            #     learningRate = float('0.' + learningRate)
            #     if model_name == 'simpleblackscholes': runningModel = mlModelsClass.BlackScholesModel_Simple(numNeurons)
            #     elif model_name == 'simpleblackscholes2layer': runningModel = mlModelsClass.BlackScholesModel_Simple2Layer(numNeurons)
            #     elif model_name == 'simpleblackscholes3layer': runningModel = mlModelsClass.BlackScholesModel_Simple3Layer(numNeurons)
            #     elif model_name == 'simpleblackscholes4layer': runningModel = mlModelsClass.BlackScholesModel_Simple4Layer(numNeurons)
            #     elif model_name == 'simpleblackscholesgreeks': runningModel = mlModelsClass.BlackScholesModel_Simple_Greeks(numNeurons)
                
            if not model: 
                print('invalid model name passed')
                return
            if(trainOrTestMode == 1):
                runningModel.load_state_dict(torch.load("models/" + model))




            # test model and get output
            if trainOrTestMode == 1 or trainOrTestMode == 2:
                criterion = nn.MSELoss()
                pred, error = testModel(runningModel, criterion, X_test, y_test)
                models_error.append([model,error])
                # Then results need to be saved somewhere so i can compare all models
                now = datetime.now()
                
                # # Save plots to an output dir, title model name and date/time ran
                if not os.path.exists('model_output'):
                    os.makedirs('model_output')
                if not os.path.exists('model_output/' + model):
                    os.makedirs('model_output/' + model)
                if not os.path.exists('model_output/' + model + '/' + now.strftime("%d_%m_%Y_%H_%M_%S")):
                    os.makedirs('model_output/' + model + '/' + now.strftime("%d_%m_%Y_%H_%M_%S"))
                
                # xAxis = []
                # for n in range(len(pred)):
                #     xAxis.append(X_test[n][1])
                    
                # figure out this straight line issue, its because of the batching i think.
                # AM i mutatating data instead of copying it I am not sure.
                # just lower epochs so I can test it way quicker
                # and then do them individually

                # plt.clf()
                # plt.scatter(xAxis, pred[0:len(pred)],marker=".", label="ml prediction", color='red')
                # plt.scatter(xAxis, y_test.float()[0:len(pred)],marker=".", label="y_test", color='black')
                # plt.xlabel('Time To Maturity')
                # plt.ylabel('Option Price')
                # plt.legend()
                # plt.savefig('model_output/' + model + "/" + now.strftime("%d_%m_%Y_%H_%M_%S") + "/" + str(learningRate) + "," + str(numNeurons) + "vsMaturity" + '.png')
                plt.clf()
                # plt.scatter(y_test.float()[0:len(pred)], pred[0:len(pred)],marker=".")
                # plt.xlabel('Test Option Price')
                # plt.ylabel('Predicted Option Price')
                # plt.savefig('model_output/' + model + "/" +  now.strftime("%d_%m_%Y_%H_%M_%S") + "/" + str(learningRate) + "," +str(numNeurons) + "testvsprediction" + '.png')
                print(pred)
                predIv, predDelta, predGamma, predRho, predTheta, predVega, testIv, testDelta, testGamma, testRho, testTheta, testVega = list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list()
                for i in range(len(pred)):
                    predIv = pred[i][0]
                    testIv = y_test[i][0]
                    predDelta = pred[i][1]
                    testDelta = y_test[i][1]
                    predGamma = pred[i][2]
                    testGamma = y_test[i][2]
                    predRho = pred[i][3]
                    testRho = y_test[i][3]
                    predTheta = pred[i][4]
                    testTheta = y_test[i][4]
                    predVega = pred[i][5]
                    testVega = y_test[i][5]
                
                fig, axs = plt.subplots(2, 3)
                    
                axs[0][0].plot(testIv, predIv)
                axs[0][1].plot(testDelta, predDelta)
                axs[0][2].plot(testGamma, predGamma)
                axs[1][0].plot(testRho, predRho)
                axs[1][1].plot(testTheta, predTheta)
                axs[1][2].plot(testVega, predVega)

                plt.show()
                
                # doesnt work wtf, try plotting individual ones i guess?
                
                plt.clf()
              
    # Then sort lowest to highest and display
    models_error.sort(key = lambda x: x[1])
                
    # write to file list of models and their mse
    f = open("model_output/modelMse.csv", "w")
    f.write("model,mse\n")
    for i in range(len(models_error)):
        f.write(models_error[i][0] + "," + str(models_error[i][1]) + "\n")
    f.close()
    # show model with lowest mse
    
    # write to a separate file model with the lowest mse
    f = open("model_output/lowesterrormodel.", "w")
    f.write(models_error[0][0] + "," + str(models_error[0][1]))
    f.close()           
       
       
       
            
