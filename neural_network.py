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
neural_network()
