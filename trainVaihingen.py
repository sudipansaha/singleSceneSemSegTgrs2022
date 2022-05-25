# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Sudipan Saha
"""
import os
import sys
import glob
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import cv2
import tifffile 
from math import floor, ceil, sqrt, exp
import time
import argparse


from networksVaryingKernel import Model4ChannelInitialToMiddleLayersDifferent,Model5ChannelInitialToMiddleLayersDifferent,Model6ChannelInitialToMiddleLayersDifferent


## Defining nanVar
nanVar=float('nan')


##Defining Parameters
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--manualSeed', type=int, default=85, help='manual seed')
parser.add_argument('--nFeaturesIntermediateLayers', type=int, default=64, help='Number of features in the intermediate layers')
parser.add_argument('--nFeaturesFinalLayer', type=int, default=8, help='Number of features in the final classification layer')
parser.add_argument('--numTrainingEpochs', type=int, default=2, help='Number of training epochs')
parser.add_argument('--modelName', type=str, default='Model5ChannelInitialToMiddleLayersDifferent', help='Model name')
opt = parser.parse_args()


##Defining training image data path
imagePath = '../data/Vaihingen/img/top_mosaic_09cm_area1.tif'
inputImage = tifffile.imread(imagePath)


##inputImage = inputImage[:,:,0:3] ###taking only RGB

##Since images are in 0-255 range, dividing by 255
inputImage = inputImage/255.

##setting manual seeds
manualSeed=opt.manualSeed
print('Manual seed is '+str(manualSeed))
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
np.random.seed(manualSeed)

modelInputMean=0
baseImageFileName = (os.path.basename(imagePath).rsplit(".",1))[0]
saveModelPath = './trainedModels/'+baseImageFileName+'.pth'

data=np.copy(inputImage)  


### Paramsters related to the CNN model
modelInputMean=0
trainingBatchSize = 8
nFeaturesIntermediateLayers = opt.nFeaturesIntermediateLayers  ##number of features in the intermediate layers
nFeaturesFinalLayer = opt.nFeaturesFinalLayer ##number of features of final classification layer
numTrainingEpochs = opt.numTrainingEpochs
maxIter = 50  ##number of maximum iterations over same batch
lr = 0.001 ##Learning rate
numberOfImageChannels = data.shape[2]
trainingPatchSize = 224 ##Patch size used for training self-sup learning
trainingStrideSize = int(trainingPatchSize)
useCuda = torch.cuda.is_available()  

##Model name
modelName = opt.modelName





palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d





    
class TrainingDatasetLoader(torch.utils.data.Dataset):
##loads data for self-supervised model learning
    def __init__(self, data, useCuda, patchSize = 112, stride = None, transform=None):
          #Initialization
          self.transform = transform
          
          ##Torchvision data transforms
          self.GaussianBlur = transforms.GaussianBlur(5, sigma=(0.1, 2.0))

          # basics
          self.transform = transform
          self.patchSize = patchSize
          if not stride:
            self.stride = 1
          else:
            self.stride = stride
          
          ##Converting from Row*Col*Channel format to Channle*Row*Col
          data = np.transpose(data, (2, 0, 1))
          
          self.data = torch.from_numpy(data)
          self.useCuda = useCuda
          self.data = self.data.type(torch.FloatTensor)
          
          
          # calculate the number of patches
          s = self.data.shape
          n1 = ceil((s[1] - self.patchSize + 1) / self.stride)
          n2 = ceil((s[2] - self.patchSize + 1) / self.stride)
          n_patches_i = n1 * n2
          self.n_patches = n_patches_i
          
          self.patch_coords = []          
          # generate path coordinates
          for i in range(n1):
                for j in range(n2):
                    # coordinates in (x1, x2, y1, y2)
                    current_patch_coords = ( 
                                    [self.stride*i, self.stride*i + self.patchSize, self.stride*j, self.stride*j + self.patchSize],
                                    [self.stride*(i + 1), self.stride*(j + 1)])
                    self.patch_coords.append(current_patch_coords)
  
    def __len__(self):
          #Denotes the total number of samples of training dataset
          return self.n_patches
  
    def __getitem__(self, idx):
          current_patch_coords = self.patch_coords[idx]
          limits = current_patch_coords[0]
          
          I1 = self.data[:, limits[0]:limits[1], limits[2]:limits[3]]
          randomTransformation = torch.randint(low=0,high=2,size=(1,)) ##here high is one above the highest integer to be drawn from the distribution.
          if randomTransformation == 0:
             I2 = self.GaussianBlur(I1)
          elif randomTransformation == 1:
             I2 = I1.clone()
             I2[1,:,:]=0
          sample = {'I1': I1,'I2': I2}
        
          if self.transform:
            sample = self.transform(sample)

          return sample




 



# train
if modelName=='Model4ChannelInitialToMiddleLayersDifferent':
    model = Model4ChannelInitialToMiddleLayersDifferent(numberOfImageChannels,nFeaturesIntermediateLayers,nFeaturesFinalLayer) 
elif modelName=='Model5ChannelInitialToMiddleLayersDifferent':
    model = Model5ChannelInitialToMiddleLayersDifferent(numberOfImageChannels,nFeaturesIntermediateLayers,nFeaturesFinalLayer)
elif modelName=='Model6ChannelInitialToMiddleLayersDifferent':
    model = Model6ChannelInitialToMiddleLayersDifferent(numberOfImageChannels,nFeaturesIntermediateLayers,nFeaturesFinalLayer)
else:
    sys.exit('Unrecognized model name')
#print(model)

if useCuda:
    model.cuda()

model.train()


lossFunction = torch.nn.CrossEntropyLoss()
lossFunctionSecondary = torch.nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) ##Adam or SGD
#optimizer = optim.Adam(model.parameters(), lr=lr)  ##Adam or SGD


  
trainingDataset = TrainingDatasetLoader(data, useCuda, patchSize = trainingPatchSize, stride = trainingStrideSize, transform=None)
trainLoader = torch.utils.data.DataLoader(dataset=trainingDataset,batch_size=trainingBatchSize,shuffle=True) 

  
    
lossPrimary1Array = torch.empty((1))
lossPrimary2Array = torch.empty((1))
lossPrimaryArray = torch.empty((1))
lossSecondary1Array = torch.empty((1))
lossSecondary2Array = torch.empty((1))
lossTotalArray = torch.empty((1))


startTime = time.time()
for epochIter in range(numTrainingEpochs):
    for batchStep, batchData in enumerate(trainLoader):
                    
        data1ForModelTraining = batchData['I1'].float().cuda()
        
        data2ForModelTraining = batchData['I2'].float().cuda()
        
        randomShufflingIndices = torch.randperm(data2ForModelTraining.shape[0])
        data2ForModelTrainingShuffled = data2ForModelTraining[randomShufflingIndices,:,:,:]
            
                
        thisBatchLoss = torch.empty((maxIter,1))
        for trainingInsideIter in range(maxIter):
            optimizer.zero_grad()
            
            projection1, projection2 = model(data1ForModelTraining, data2ForModelTraining)
            _,projection2Shuffled = model(data1ForModelTraining,data2ForModelTrainingShuffled)
            
            _,prediction1 = torch.max(projection1,1)
            _,prediction2 = torch.max(projection2,1)
            _,prediction2Shuffled = torch.max(projection2Shuffled,1)
            
            lossPrimary1 = lossFunction(projection1, prediction1) 
            lossPrimary2 = lossFunction(projection2, prediction2)
            
            lossSecondary1 = lossFunctionSecondary(projection1,projection2)
            lossSecondary2 = -lossFunctionSecondary(projection1,projection2Shuffled)
            
            lossPrimary = (lossPrimary1+lossPrimary2)/2
            lossTotal = (lossPrimary1+lossPrimary2+lossSecondary1+lossSecondary2)/4
            #lossTotal = (lossPrimary1+lossPrimary2+lossSecondary1)/3
            
            lossPrimary1Array = torch.cat((lossPrimary1Array,lossPrimary1.unsqueeze(0).cpu().detach()))
            lossPrimary2Array = torch.cat((lossPrimary2Array,lossPrimary2.unsqueeze(0).cpu().detach()))
            lossPrimaryArray = torch.cat((lossPrimaryArray,lossPrimary.unsqueeze(0).cpu().detach()))
            lossSecondary1Array = torch.cat((lossSecondary1Array,lossSecondary1.unsqueeze(0).cpu().detach()))
            lossSecondary2Array = torch.cat((lossSecondary2Array,lossSecondary2.unsqueeze(0).cpu().detach()))
            lossTotalArray = torch.cat((lossTotalArray,lossTotal.unsqueeze(0).cpu().detach()))
            
            
                        
            if epochIter==0:
                lossPrimary.backward()
            else:
                lossTotal.backward()

            optimizer.step()
        #print ('Epoch: ',epochIter, '/', numTrainingEpochs, 'batch: ',batchStep)
    #print('End of epoch', epochIter)
torch.save(model,saveModelPath)    

###There is an extra zero in loss arrays at beginning
lossPrimary1Array = lossPrimary1Array[1:-1] 
lossPrimary2Array = lossPrimary2Array[1:-1] 
lossPrimayArray = lossPrimaryArray[1:-1]
lossSecondary1Array = lossSecondary1Array[1:-1] 
lossSecondary2Array = lossSecondary2Array[1:-1] 
lossTotalArray = lossTotalArray[1:-1] 













    
    
    
    
    






    



