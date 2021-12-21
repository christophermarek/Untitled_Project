import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
from torch import nn, optim
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
# local file
import models as mlModelsClass
import data_generator


# https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=2513&context=gradreports
# It contains code too I didnt realize this before, shows how they generate their data exactly

# 6 outputs for 1 greek but thats later.

# ADD to readme explanation of each file and directory


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

    # Sort by time to maturity so model has sense of time
    df = df.sort_values(by=['timetomaturity'], ascending=True)
    print(df.head())
    
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



def trainModel(model, optimizer, lossFN, input, output, numEpochs):
    model.train()
    # begin to train
    for i in range(15):
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
    
    models_error = list()
    
    for model in models:
        for hyperParamConfigEntry in hyperparam_config:
            learningRate = hyperParamConfigEntry[0]
            numNeurons = hyperParamConfigEntry[1]

            print('running model: ' + model + " with config: " + str(learningRate) + " | " + str(numNeurons)) 
            
            # build the model
            runningModel = False
            if not trainOrTestMode == 1:
                if model == 'simpleblackscholes': runningModel = mlModelsClass.BlackScholesModel_Simple(numNeurons)
                elif model == 'simpleblackscholes2layer': runningModel = mlModelsClass.BlackScholesModel_Simple2Layer(numNeurons)
                elif model == 'simpleblackscholes3layer': runningModel = mlModelsClass.BlackScholesModel_Simple3Layer(numNeurons)
                elif model == 'simpleblackscholes4layer': runningModel = mlModelsClass.BlackScholesModel_Simple4Layer(numNeurons)
            else:
                print(model)
                modelSplit = model.split(',')
                model_name = modelSplit[0]
                restOfModel = ''.join(modelSplit[1:])
                restOfModelSplit = restOfModel.split('.')
                numNeurons = int(restOfModelSplit[1][-2:])
                learningRate = restOfModelSplit[1][:-2]
                learningRate = float('0.' + learningRate)
                if model_name == 'simpleblackscholes': runningModel = mlModelsClass.BlackScholesModel_Simple(numNeurons)
                elif model_name == 'simpleblackscholes2layer': runningModel = mlModelsClass.BlackScholesModel_Simple2Layer(numNeurons)
                elif model_name == 'simpleblackscholes3layer': runningModel = mlModelsClass.BlackScholesModel_Simple3Layer(numNeurons)
                elif model_name == 'simpleblackscholes4layer': runningModel = mlModelsClass.BlackScholesModel_Simple4Layer(numNeurons)
                
            if not model: 
                print('invalid model name passed')
                return
            if(trainOrTestMode == 1):
                runningModel.load_state_dict(torch.load("models/" + model))

            criterion = nn.MSELoss()

            # use LBFGS as optimizer since we can load the whole data to train
            # lr = learning rate
            # Will need to test different learning rates and optimizers
            optimizer = optim.LBFGS(runningModel.parameters(), lr=learningRate)

            # train model
            if trainOrTestMode == 0 or trainOrTestMode == 2:
                trainModel(runningModel, optimizer, criterion, X_train, y_train, 1)
                if not os.path.exists('models'):
                    os.makedirs('models')
                torch.save(runningModel.state_dict(), "models/" + model + "," + str(learningRate) + str(numNeurons) + '.ckpt')

            # test model and get output
            if trainOrTestMode == 1 or trainOrTestMode == 2:
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
                
                xAxis = []
                for n in range(len(pred)):
                    xAxis.append(X_test[n][1])
                    
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
                plt.scatter(y_test.float()[0:len(pred)], pred[0:len(pred)],marker=".")
                plt.xlabel('Test Option Price')
                plt.ylabel('Predicted Option Price')
                plt.savefig('model_output/' + model + "/" +  now.strftime("%d_%m_%Y_%H_%M_%S") + "/" + str(learningRate) + "," +str(numNeurons) + "testvsprediction" + '.png')
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
       
            
def preRun():
    # get params
    # mode = input("Enter 2 for train and test, 1 for test mode, 0 for train mode:  ")
    # mode = int(mode)
    # if not (mode in [0,1,2]):
    #     print('invalid mode paramaters')
    #     return
    
    # if not mode == 1:
    #     models = [
    #         "simpleblackscholes",
    #         "simpleblackscholes2layer",
    #         "simpleblackscholes3layer",
    #         "simpleblackscholes4layer"
    #     ]
        
    #     lr = [0.1]
    #     hiddenLayer = [10]
    #     hyperparam_config = list()
    #     for rate in lr:
    #         for neuronCount in hiddenLayer:
    #             hyperparam_config.append([rate, neuronCount])
                
    # else:
    #     modelsDir = os.listdir('models')
    #     # print(modelsDir)
    #     models = modelsDir
    #     # will never be ran with this config but it means 1 iteration per model
    #     hyperparam_config = [[0.1, 10]]
    
    generatingData = True    
    dataset_title = 'blackscholesprices_and_greeks'
    if generatingData:
        dataGenerator = data_generator.DataGenerator(dataset_title)
        
        output = dataGenerator.generateDataSet(5000000)
        # Class supports additional outputs with changing input ranges to generate multiple datasets
        
        print(output[1])
        if not output[0]:
            return

        print("\n")

        
    dataSetPath = 'generated_datasets/' + dataset_title + '.csv'
    dataSet = loadData(dataSetPath)
    if not dataSet: 
        print('invalid dataset path')
        return
    
    # hyperparamatertesting config.
    # lr = [0.01, 0.1, 0.3, 0.5, 0.8]
    # main(mode, models, dataSet, hyperparam_config)

preRun()
