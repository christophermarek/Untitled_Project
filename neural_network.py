import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import pandas as pd


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

    # wonder if i should round output price column, we will see later

    # Dataset needs an output column
    
    # # transforms.Normalize() — normalizes the tensor with a mean and standard deviation which goes as the two parameters respectively.
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.5,), (0.5,)),
    #                                 ])

    # # Now we finally download the data sets, shuffle them and transform each of them. We download the data sets and load them to DataLoader,
    # # which combines the data-set and a sampler and provides single- or multi-process iterators over the data-set.
    # This datasets.MNIST needs to be replace with whats in lecture
    # trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET',
    #                           download=True, train=True, transform=transform)
    # valset = datasets.MNIST('PATH_TO_STORE_TESTSET',
    #                         download=True, train=False, transform=transform)
    # # In one line, batch size is the number of images we want to read in one go.
    # trainloader = torch.utils.data.DataLoader(
    #     trainset, batch_size=64, shuffle=True)
    # valloader = torch.utils.data.DataLoader(
    #     valset, batch_size=64, shuffle=True)

    # # Visualize data
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()

    # print(images.shape)
    # print(labels.shape)

    # figure = plt.figure()

    # num_of_images = 60
    # for index in range(1, num_of_images + 1):
    #     plt.subplot(6, 10, index)
    #     plt.axis('off')
    #     plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

    # # random sample of images in trainingset
    # plt.show()

neural_network()
