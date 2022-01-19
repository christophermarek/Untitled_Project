
from torch import nn, optim
import os
import torch
from datetime import datetime
from time import time
import matplotlib.pyplot as plt

# local imports
import data_generator
import models as mlModelsClass

from neural_network import loadData, trainModel, testModel


def get_model(model_name):
    runningModel = False
    if model_name == 'BlackScholesModel_Simple_Greeks_1layer':
        runningModel = mlModelsClass.BlackScholesModel_Simple_Greeks_1layer()
    elif model_name == 'BlackScholesModel_Simple_Greeks_2layer':
        runningModel = mlModelsClass.BlackScholesModel_Simple_Greeks_2layer()
    elif model_name == 'BlackScholesModel_Simple_Greeks_4layer':
        runningModel = mlModelsClass.BlackScholesModel_Simple_Greeks_4layer()
        
    elif model_name == 'BS_DELTA_SIMPLE':
        runningModel = mlModelsClass.BS_DELTA_SIMPLE()
    elif model_name == 'BS_GAMMA_SIMPLE':
        runningModel = mlModelsClass.BS_GAMMA_SIMPLE()
    elif model_name == 'BS_RHO_SIMPLE':
        runningModel = mlModelsClass.BS_RHO_SIMPLE()
    elif model_name == 'BS_THETA_SIMPLE':
        runningModel = mlModelsClass.BS_THETA_SIMPLE()
    elif model_name == 'BS_VEGA_SIMPLE':
        runningModel = mlModelsClass.BS_VEGA_SIMPLE()
    elif model_name == 'BS_DELTA_SIMPLE2':
        runningModel = mlModelsClass.BS_DELTA_SIMPLE2()
    elif model_name == 'BS_GAMMA_SIMPLE2':
        runningModel = mlModelsClass.BS_GAMMA_SIMPLE2()
    elif model_name == 'BS_RHO_SIMPLE2':
        runningModel = mlModelsClass.BS_RHO_SIMPLE2()
    elif model_name == 'BS_THETA_SIMPLE2':
        runningModel = mlModelsClass.BS_THETA_SIMPLE2()
    elif model_name == 'BS_VEGA_SIMPLE2':
        runningModel = mlModelsClass.BS_VEGA_SIMPLE2()
    return runningModel


def instruction_generate_dataset(input):
    print('INSTRUCTION: Generate Dataset')

    dataset_title = input[0]
    dataset_size = int(input[1])

    dataGenerator = data_generator.DataGenerator(dataset_title)
    output = dataGenerator.generateDataSet(dataset_size)

    print(output[1])
    if not output[0]:
        return

    print('INSTRUCTION COMPLETE: Generate Dataset \n')


def instruction_train_model(input):
    print('INSTRUCTION Train Model')

    dataset_path = input[0]
    model_name = input[1]
    model_init_params = input[2]
    learning_rate = float(input[3])
    num_epochs = int(input[4])
    output_columns = input[5].split(',')

    # not a config entry yet
    criterion = nn.MSELoss()

    dataSet = loadData(dataset_path, output_columns)
    if not dataSet:
        print('invalid dataset path')
        return

    X_train = dataSet[0]
    y_train = dataSet[2]

    runningModel = get_model(model_name)
    if not runningModel:
        print('INSTRUCTION FAILED: Train Model')
        return

    # use LBFGS as optimizer since we can load the whole data to train
    # lr = learning rate
    # Will need to test different learning rates and optimizers
    optimizer = optim.LBFGS(runningModel.parameters(), lr=learning_rate)

    # train model
    trainModel(runningModel, optimizer, criterion,
               X_train, y_train, num_epochs)
    if not os.path.exists('models'):
        os.makedirs('models')
    # will overwrite existing trained model
    # can leave this code in the train model in nn to remove imports from this file
    torch.save(runningModel.state_dict(), "models/" + model_name + '.ckpt')

    print('INSTRUCTION Complete: Train Model \n')


def instruction_test_model(input):
    print('INSTRUCTION TEST Model')

    dataset_path = input[0]
    model_name = input[1]
    model_path = input[2]
    percent_dataset = input[3] # unused
    output_columns = input[4].split(',')
    

    print('\n MODEL: ' + model_name + '\n')

    dataSet = loadData(dataset_path, output_columns)
    if not dataSet:
        print('INSTRUCTION FAILED: TEST MODEL, invalid dataset path')
        return

    X_test = dataSet[1]
    y_test = dataSet[3]

    runningModel = get_model(model_name)
    if not runningModel:
        print('INSTRUCTION FAILED: TEST MODEL, invalid model_name')
        return
    runningModel.load_state_dict(torch.load("models/" + model_path))

    criterion = nn.MSELoss()

    
    pred, error = testModel(runningModel, criterion, X_test, y_test)

    print('INSTRUCTION Complete: TEST Model \n')

    return [pred, error]


def instruction_output_model(input):

    print('INSTRUCTION OUTPUT Model')

    pred, error = instruction_test_model(input)

    dataset_path = input[0]
    model_name = input[1]
    model_path = input[2]
    percent_dataset = float(input[3])

    dataSet = loadData(dataset_path)
    if not dataSet:
        print('INSTRUCTION FAILED: TEST MODEL, invalid dataset path')
        return
    y_test = dataSet[3]

    now = datetime.now()

    # # Save plots to an output dir, title model name and date/time ran
    if not os.path.exists('model_output'):
        os.makedirs('model_output')
    if not os.path.exists('model_output/' + model_name):
        os.makedirs('model_output/' + model_name)
        
    
    # THIS IS REALLY INNEFFICIENT, CANT I PLOT IN THE SAME LOOP? I ALREADY KNOW SIZE OF X/Y
    predDelta, predGamma, predRho, predTheta, predVega, testDelta, testGamma, testRho, testTheta, testVega = list(
    ), list(), list(), list(), list(), list(), list(), list(), list(), list()
    for i in range(round(len(pred) * percent_dataset)):
        predDelta.append(pred[i][0])
        testDelta.append(y_test[i][0])
        predGamma.append(pred[i][1])
        testGamma.append(y_test[i][1])
        predRho.append(pred[i][2])
        testRho.append(y_test[i][2])
        predTheta.append(pred[i][3])
        testTheta.append(y_test[i][3])
        predVega.append(pred[i][4])
        testVega.append(y_test[i][4])

    # plt.clf()
    # plt.scatter(testIv, predIv, marker=".", label="ml prediction", color='red')
    # plt.xlabel('Test IV')
    # plt.ylabel('Prediction IV')
    # plt.savefig('model_output/' + model_name + "/IV" + '.png')
    plt.clf()
    plt.scatter(testDelta, predDelta, marker=".", label="ml prediction", color='red')
    plt.xlabel('Test DELTA')
    plt.ylabel('Prediction DELTA')
    plt.savefig('model_output/' + model_name + "/DELTA" + '.png')
    plt.clf()
    plt.scatter(testGamma, predGamma, marker=".", label="ml prediction", color='red')
    plt.xlabel('Test GAMMA')
    plt.ylabel('Prediction GAMMA')
    plt.savefig('model_output/' + model_name + "/GAMMA" + '.png')
    plt.clf()
    plt.scatter(testRho, predRho, marker=".", label="ml prediction", color='red')
    plt.xlabel('Test RHO')
    plt.ylabel('Prediction RHO')
    plt.savefig('model_output/' + model_name + "/RHO" + '.png')
    plt.clf()
    plt.scatter(testTheta, predTheta, marker=".", label="ml prediction", color='red')
    plt.xlabel('Test THETA')
    plt.ylabel('Prediction THETA')
    plt.savefig('model_output/' + model_name + "/THETA" + '.png')
    plt.clf()
    plt.scatter(testVega, predVega, marker=".", label="ml prediction", color='red')
    plt.xlabel('Test VEGA')
    plt.ylabel('Prediction VEGA')
    plt.savefig('model_output/' + model_name + "/VEGA" + '.png')
    
    print('INSTRUCTION Complete: OUTPUT Model \n')

def instruction_greek_model(input):

    print('INSTRUCTION OUTPUT Model')

    pred, error = instruction_test_model(input)

    dataset_path = input[0]
    model_name = input[1]
    model_path = input[2]
    percent_dataset = float(input[3])
    output_column_name = input[4]

    dataSet = loadData(dataset_path, output_column_name)
    if not dataSet:
        print('INSTRUCTION FAILED: TEST MODEL, invalid dataset path')
        return
    y_test = dataSet[3]

    now = datetime.now()

    # # Save plots to an output dir, title model name and date/time ran
    if not os.path.exists('model_output'):
        os.makedirs('model_output')
    if not os.path.exists('model_output/' + model_name):
        os.makedirs('model_output/' + model_name)
        
    
    # print(pred)
    
    # THIS IS REALLY INNEFFICIENT, CANT I PLOT IN THE SAME LOOP? I ALREADY KNOW SIZE OF X/Y
    # output, pred = list(), list()
    # for i in range(round(len(pred) * percent_dataset)):
    #     predDelta.append(pred[i][0])
    #     testDelta.append(y_test[i][0])
    #     predGamma.append(pred[i][1])
    #     testGamma.append(y_test[i][1])
    #     predRho.append(pred[i][2])
    #     testRho.append(y_test[i][2])
    #     predTheta.append(pred[i][3])
    #     testTheta.append(y_test[i][3])
    #     predVega.append(pred[i][4])
    #     testVega.append(y_test[i][4])

    plt.clf()
    plt.scatter(y_test, pred, marker=".", label="ml prediction", color='red')
    plt.title('MSE: ' + str(error))
    plt.xlabel('Test ' + output_column_name)
    plt.ylabel('Prediction DELTA ' + output_column_name )
    plt.savefig('model_output/' + model_name + "/" + output_column_name + '.png')
    plt.clf()

    
    print('INSTRUCTION Complete: OUTPUT Model \n')

def main():

    print("Program Started \n")

    print("Loading Config \n")
    path_to_config = 'config.txt'
    if not path_to_config:
        print('invalid path to config')
        return

    print("Reading Config \n")
    with open(path_to_config) as file:

        inComment = False
        inInstruction = False
        capturedLines = list()
        processFunction = None

        for line in file:
            if line.strip() == '':
                continue

            if line.strip() == '!':
                inComment = not inComment
                continue

            if not inComment:
                if not inInstruction:
                    # Capture instruction type
                    if line.strip() == 'GENERATEDATASET':
                        processFunction = instruction_generate_dataset
                        inInstruction = True
                    if line.strip() == 'TRAIN':
                        processFunction = instruction_train_model
                        inInstruction = True
                    if line.strip() == 'TEST':
                        processFunction = instruction_test_model
                        inInstruction = True
                    if line.strip() == 'OUTPUT':
                        processFunction = instruction_output_model
                        inInstruction = True
                    if line.strip() == 'OUTPUT_SINGLE_GREEK':
                        processFunction = instruction_greek_model
                        inInstruction = True 

                else:
                    if line.strip() == 'END':
                        processFunction(capturedLines)
                        inInstruction = False
                        processFunciton = None
                        capturedLines = list()
                    else:
                        capturedLines.append(line.strip())

    print("Completed Going Through Config \n")


if __name__ == "__main__":
    main()
