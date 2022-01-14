
from torch import nn, optim
import os
import torch

# local imports
import data_generator
import models as mlModelsClass

from neural_network import loadData, trainModel, testModel


def get_model(model_name):
    runningModel = False
    if model_name == 'BlackScholesModel_Simple_Greeks':
        runningModel = mlModelsClass.BlackScholesModel_Simple_Greeks()
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

    # not a config entry yet
    criterion = nn.MSELoss()

    dataSet = loadData(dataset_path)
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

    print('\n MODEL: ' + model_name + '\n')

    dataSet = loadData(dataset_path)
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


def instruction_output_model(input):
    now = datetime.now()

       # # Save plots to an output dir, title model name and date/time ran
       if not os.path.exists('model_output'):
            os.makedirs('model_output')
        if not os.path.exists('model_output/' + model):
            os.makedirs('model_output/' + model)
        if not os.path.exists('model_output/' + model + '/' + now.strftime("%d_%m_%Y_%H_%M_%S")):
            os.makedirs('model_output/' + model + '/' +
                        now.strftime("%d_%m_%Y_%H_%M_%S"))

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
        
        WHY DONT I DO A SEPARATE FILE FOR EAch GREEK AT FIRST< I CAN ONLWAYS COMBINE THE INDIVIDUAL PICS LATER AND ITS EASIER
        
        print(pred)
        predIv, predDelta, predGamma, predRho, predTheta, predVega, testIv, testDelta, testGamma, testRho, testTheta, testVega = list(
        ), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list()
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


def main():

    print("Program Started \n")

    np.random.seed(0)
    torch.manual_seed(0)

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
