
from torch import nn, optim
import os
import torch

#local imports
import data_generator
import models as mlModelsClass

from neural_network import loadData, trainModel


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
    
    # load data into training set, maybe move into main?
    dataSet = loadData(dataset_path)
    if not dataSet: 
        print('invalid dataset path')
        return
    
    X_train = dataSet[0]
    y_train = dataSet[2]
    
    runningModel = False
    if model_name == 'BlackScholesModel_Simple_Greeks': 
        runningModel = mlModelsClass.BlackScholesModel_Simple_Greeks()
        
    if not runningModel:
        print('INSTRUCTION FAILED: Train Model')
        return 
    
    # use LBFGS as optimizer since we can load the whole data to train
    # lr = learning rate
    # Will need to test different learning rates and optimizers
    optimizer = optim.LBFGS(runningModel.parameters(), lr=learning_rate)

    # train model
    trainModel(runningModel, optimizer, criterion, X_train, y_train, num_epochs)
    if not os.path.exists('models'):
        os.makedirs('models')
    # will overwrite existing trained model
    # can leave this code in the train model in nn to remove imports from this file
    torch.save(runningModel.state_dict(), "models/" + 'model_name'  + '.ckpt')
    
    print('INSTRUCTION Complete: Train Model \n')    
    
    
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