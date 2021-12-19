import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def configureOutput():
    rootDir = './model_output'
    models = os.listdir(rootDir)
    
    for model in models:
        imageDirs = os.listdir(rootDir + '/' + model)
        
        imageArray = list()
        pathToImage = rootDir + '/' + model + '/'
        if len(imageDirs) == 15:
            for i in range(0, len(imageDirs), 3):
                imageArray.append([mpimg.imread(pathToImage + imageDirs[i] + '/' + os.listdir(pathToImage + imageDirs[i])[0]) , mpimg.imread(pathToImage + imageDirs[i+1] + '/' + os.listdir(pathToImage + imageDirs[i+1])[0]), mpimg.imread(pathToImage + imageDirs[i+2] + '/' + os.listdir(pathToImage + imageDirs[i+2])[0])])
            # Now each model has an image array so now we plot for this model
            fig, axs = plt.subplots(5, 3)
            fig.suptitle(model, fontsize=16)
            # fig.supxlabel('Hidden Layer Neuron Count [10, 25, 50]')
            # fig.supylabel('Learning Rates [0.01, 0.1, 0.3, 0.5, 0.8]')

            for n in range(len(imageArray)):
                axs[n][0].imshow(imageArray[n][0], aspect='auto')
                axs[n][1].imshow(imageArray[n][1], aspect='auto')
                axs[n][2].imshow(imageArray[n][2], aspect='auto')
                axs[n][0].axis('off')
                axs[n][1].axis('off')
                axs[n][2].axis('off')
                
            fig.subplots_adjust(top=0.88)
            plt.savefig('model_output/' + model + '_combinedimages.pdf', dpi=1200)
            print('fig saved')
            
    
configureOutput()