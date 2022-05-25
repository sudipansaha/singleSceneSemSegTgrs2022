# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Sudipan Saha
"""
import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio






# CNN model 5 channel
class Model5ChannelInitialToMiddleLayersDifferent(nn.Module):
    def __init__(self,numberOfImageChannels,nFeaturesIntermediateLayers,nFeaturesFinalLayer):
        super(Model5ChannelInitialToMiddleLayersDifferent, self).__init__()
        
        ##see this page: https://discuss.pytorch.org/t/extracting-and-using-features-from-a-pretrained-model/20723
        #pretrainedModel = vgg16(pretrained=True)
        #self.Vgg16features = nn.Sequential(
         #           *list(pretrainedModel.features.children())[:3])
        
        kernelSize=3
        paddingSize=int((kernelSize-1)/2)
        
        ##Conv layer 1
        self.conv1Modality1 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize )
        self.bn1Modality1 = nn.BatchNorm2d(nFeaturesIntermediateLayers)
        self.conv1Modality1.weight=torch.nn.init.kaiming_uniform_(self.conv1Modality1.weight)
        self.conv1Modality1.bias.data.fill_(0.01)
        #self.bn1.weight.data.fill_(1)
        self.bn1Modality1.bias.data.fill_(0.001)
        
        
        self.conv1Modality2 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize )
        self.bn1Modality2 = nn.BatchNorm2d(nFeaturesIntermediateLayers)
        self.conv1Modality2.weight=torch.nn.init.kaiming_uniform_(self.conv1Modality2.weight)
        self.conv1Modality2.bias.data.fill_(0.01)
        #self.bn1.weight.data.fill_(1)
        self.bn1Modality2.bias.data.fill_(0.001)
        
        ##Conv layer 2
        self.conv2Modality1 = nn.Conv2d(nFeaturesIntermediateLayers, nFeaturesIntermediateLayers*2, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn2Modality1 = nn.BatchNorm2d(nFeaturesIntermediateLayers*2) 
        self.conv2Modality1.weight=torch.nn.init.kaiming_uniform_(self.conv2Modality1.weight)
        self.conv2Modality1.bias.data.fill_(0.01)
        #self.bn2Modality1.weight.data.fill_(1)
        self.bn2Modality1.bias.data.fill_(0.001)
        
        
        self.conv2Modality2 = nn.Conv2d(nFeaturesIntermediateLayers, nFeaturesIntermediateLayers*2, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn2Modality2 = nn.BatchNorm2d(nFeaturesIntermediateLayers*2) 
        self.conv2Modality2.weight=torch.nn.init.kaiming_uniform_(self.conv2Modality2.weight)
        self.conv2Modality2.bias.data.fill_(0.01)
        #self.bn2Modality1.weight.data.fill_(1)
        self.bn2Modality2.bias.data.fill_(0.001)
        
        ##Conv layer 3
        self.conv3Modality1 = nn.Conv2d(nFeaturesIntermediateLayers*2, nFeaturesIntermediateLayers*2, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn3Modality1 = nn.BatchNorm2d(nFeaturesIntermediateLayers*2)
        self.conv3Modality1.weight=torch.nn.init.kaiming_uniform_(self.conv3Modality1.weight)
        self.conv3Modality1.bias.data.fill_(0.01)
        #self.bn3Modality1.weight.data.fill_(1)
        self.bn3Modality1.bias.data.fill_(0.001)
        
        self.conv3Modality2 = nn.Conv2d(nFeaturesIntermediateLayers*2, nFeaturesIntermediateLayers*2, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn3Modality2 = nn.BatchNorm2d(nFeaturesIntermediateLayers*2)
        self.conv3Modality2.weight=torch.nn.init.kaiming_uniform_(self.conv3Modality2.weight)
        self.conv3Modality2.bias.data.fill_(0.01)
        #self.bn3Modality1.weight.data.fill_(1)
        self.bn3Modality2.bias.data.fill_(0.001)
        
        
        
        ##Conv layer 4
        self.conv4Modality1 = nn.Conv2d(nFeaturesIntermediateLayers*2, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn4Modality1 = nn.BatchNorm2d(nFeaturesIntermediateLayers)
        self.conv4Modality1.weight=torch.nn.init.kaiming_uniform_(self.conv4Modality1.weight)
        self.conv4Modality1.bias.data.fill_(0.01)
        #self.bn3Modality1.weight.data.fill_(1)
        self.bn4Modality1.bias.data.fill_(0.001)
        
        
        self.conv4Modality2 = nn.Conv2d(nFeaturesIntermediateLayers*2, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn4Modality2 = nn.BatchNorm2d(nFeaturesIntermediateLayers)
        self.conv4Modality2.weight=torch.nn.init.kaiming_uniform_(self.conv4Modality2.weight)
        self.conv4Modality2.bias.data.fill_(0.01)
        #self.bn4Modality1.weight.data.fill_(1)
        self.bn4Modality2.bias.data.fill_(0.001)
        
        
         
        ##Conv layer 5
        self.conv5 = nn.Conv2d(nFeaturesIntermediateLayers, nFeaturesFinalLayer, kernel_size=1, stride=1, padding=0 )
        self.bn5 = nn.BatchNorm2d(nFeaturesFinalLayer)
        self.conv5.weight=torch.nn.init.kaiming_uniform_(self.conv5.weight)
        self.conv5.bias.data.fill_(0.01)
        #self.bn5.weight.data.fill_(1)
        self.bn5.bias.data.fill_(0.001)
        

    def forward(self, xModality1, xModality2):
        #x = self.Vgg16features(x)
        xModality1 = self.conv1Modality1(xModality1)
        xModality1 = F.relu( xModality1 )
        xModality1 = self.bn1Modality1(xModality1)
        xModality1 = self.conv2Modality1(xModality1)
        xModality1 = F.relu( xModality1 )
        xModality1 = self.bn2Modality1(xModality1)
        xModality1 = self.conv3Modality1(xModality1)
        xModality1 = F.relu( xModality1 )
        xModality1 = self.bn3Modality1(xModality1)
        xModality1 = self.conv4Modality1(xModality1)
        xModality1 = F.relu( xModality1 )
        xModality1 = self.bn4Modality1(xModality1)
        xModality1 = self.conv5(xModality1)
        xModality1 = self.bn5(xModality1)  ##Note there is no relu between last conv and bn
        
        xModality2 = self.conv1Modality2(xModality2)
        xModality2 = F.relu( xModality2 )
        xModality2 = self.bn1Modality2(xModality2)
        xModality2 = self.conv2Modality2(xModality2)
        xModality2 = F.relu( xModality2 )
        xModality2 = self.bn2Modality2(xModality2)
        xModality2 = self.conv3Modality2(xModality2)
        xModality2 = F.relu( xModality2 )
        xModality2 = self.bn3Modality2(xModality2)
        xModality2 = self.conv4Modality2(xModality2)
        xModality2 = F.relu( xModality2 )
        xModality2 = self.bn4Modality2(xModality2)
        xModality2 = self.conv5(xModality2)
        xModality2 = self.bn5(xModality2)  ##Note there is no relu between last conv and bn
        
        return xModality1, xModality2
        
        
        
        
        
# CNN model 4 channel
class Model4ChannelInitialToMiddleLayersDifferent(nn.Module):
    def __init__(self,numberOfImageChannels,nFeaturesIntermediateLayers,nFeaturesFinalLayer):
        super(Model4ChannelInitialToMiddleLayersDifferent, self).__init__()
        
        ##see this page: https://discuss.pytorch.org/t/extracting-and-using-features-from-a-pretrained-model/20723
        #pretrainedModel = vgg16(pretrained=True)
        #self.Vgg16features = nn.Sequential(
         #           *list(pretrainedModel.features.children())[:3])
        
        kernelSize=3
        paddingSize=int((kernelSize-1)/2)
        
        ##Conv layer 1
        self.conv1Modality1 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize )
        self.bn1Modality1 = nn.BatchNorm2d(nFeaturesIntermediateLayers)
        self.conv1Modality1.weight=torch.nn.init.kaiming_uniform_(self.conv1Modality1.weight)
        self.conv1Modality1.bias.data.fill_(0.01)
        #self.bn1.weight.data.fill_(1)
        self.bn1Modality1.bias.data.fill_(0.001)
        
        
        self.conv1Modality2 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize )
        self.bn1Modality2 = nn.BatchNorm2d(nFeaturesIntermediateLayers)
        self.conv1Modality2.weight=torch.nn.init.kaiming_uniform_(self.conv1Modality2.weight)
        self.conv1Modality2.bias.data.fill_(0.01)
        #self.bn1.weight.data.fill_(1)
        self.bn1Modality2.bias.data.fill_(0.001)
        
        ##Conv layer 2
        self.conv2Modality1 = nn.Conv2d(nFeaturesIntermediateLayers, nFeaturesIntermediateLayers*2, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn2Modality1 = nn.BatchNorm2d(nFeaturesIntermediateLayers*2) 
        self.conv2Modality1.weight=torch.nn.init.kaiming_uniform_(self.conv2Modality1.weight)
        self.conv2Modality1.bias.data.fill_(0.01)
        #self.bn2Modality1.weight.data.fill_(1)
        self.bn2Modality1.bias.data.fill_(0.001)
        
        
        self.conv2Modality2 = nn.Conv2d(nFeaturesIntermediateLayers, nFeaturesIntermediateLayers*2, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn2Modality2 = nn.BatchNorm2d(nFeaturesIntermediateLayers*2) 
        self.conv2Modality2.weight=torch.nn.init.kaiming_uniform_(self.conv2Modality2.weight)
        self.conv2Modality2.bias.data.fill_(0.01)
        #self.bn2Modality1.weight.data.fill_(1)
        self.bn2Modality2.bias.data.fill_(0.001)
        
        ##Conv layer 3
        self.conv3Modality1 = nn.Conv2d(nFeaturesIntermediateLayers*2, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn3Modality1 = nn.BatchNorm2d(nFeaturesIntermediateLayers)
        self.conv3Modality1.weight=torch.nn.init.kaiming_uniform_(self.conv3Modality1.weight)
        self.conv3Modality1.bias.data.fill_(0.01)
        #self.bn3Modality1.weight.data.fill_(1)
        self.bn3Modality1.bias.data.fill_(0.001)
        
        self.conv3Modality2 = nn.Conv2d(nFeaturesIntermediateLayers*2, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn3Modality2 = nn.BatchNorm2d(nFeaturesIntermediateLayers)
        self.conv3Modality2.weight=torch.nn.init.kaiming_uniform_(self.conv3Modality2.weight)
        self.conv3Modality2.bias.data.fill_(0.01)
        #self.bn3Modality1.weight.data.fill_(1)
        self.bn3Modality2.bias.data.fill_(0.001)
        
        
         
        ##Conv layer 4
        self.conv4 = nn.Conv2d(nFeaturesIntermediateLayers, nFeaturesFinalLayer, kernel_size=1, stride=1, padding=0 )
        self.bn4 = nn.BatchNorm2d(nFeaturesFinalLayer)
        self.conv4.weight=torch.nn.init.kaiming_uniform_(self.conv4.weight)
        self.conv4.bias.data.fill_(0.01)
        self.bn4.bias.data.fill_(0.001)
        

    def forward(self, xModality1, xModality2):
        #x = self.Vgg16features(x)
        xModality1 = self.conv1Modality1(xModality1)
        xModality1 = F.relu( xModality1 )
        xModality1 = self.bn1Modality1(xModality1)
        xModality1 = self.conv2Modality1(xModality1)
        xModality1 = F.relu( xModality1 )
        xModality1 = self.bn2Modality1(xModality1)
        xModality1 = self.conv3Modality1(xModality1)
        xModality1 = F.relu( xModality1 )
        xModality1 = self.bn3Modality1(xModality1)
        xModality1 = self.conv4(xModality1)
        xModality1 = self.bn4(xModality1)  ##Note there is no relu between last conv and bn
        
        xModality2 = self.conv1Modality2(xModality2)
        xModality2 = F.relu( xModality2 )
        xModality2 = self.bn1Modality2(xModality2)
        xModality2 = self.conv2Modality2(xModality2)
        xModality2 = F.relu( xModality2 )
        xModality2 = self.bn2Modality2(xModality2)
        xModality2 = self.conv3Modality2(xModality2)
        xModality2 = F.relu( xModality2 )
        xModality2 = self.bn3Modality2(xModality2)
        xModality2 = self.conv4(xModality2)
        xModality2 = self.bn4(xModality2)  ##Note there is no relu between last conv and bn
        
        return xModality1, xModality2



# CNN model 6 channel
class Model6ChannelInitialToMiddleLayersDifferent(nn.Module):
    def __init__(self,numberOfImageChannels,nFeaturesIntermediateLayers,nFeaturesFinalLayer):
        super(Model6ChannelInitialToMiddleLayersDifferent, self).__init__()
        
        ##see this page: https://discuss.pytorch.org/t/extracting-and-using-features-from-a-pretrained-model/20723
        #pretrainedModel = vgg16(pretrained=True)
        #self.Vgg16features = nn.Sequential(
         #           *list(pretrainedModel.features.children())[:3])
        
        kernelSize=3
        paddingSize=int((kernelSize-1)/2)
        
        ##Conv layer 1
        self.conv1Modality1 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize )
        self.bn1Modality1 = nn.BatchNorm2d(nFeaturesIntermediateLayers)
        self.conv1Modality1.weight=torch.nn.init.kaiming_uniform_(self.conv1Modality1.weight)
        self.conv1Modality1.bias.data.fill_(0.01)
        #self.bn1.weight.data.fill_(1)
        self.bn1Modality1.bias.data.fill_(0.001)
        
        
        self.conv1Modality2 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize )
        self.bn1Modality2 = nn.BatchNorm2d(nFeaturesIntermediateLayers)
        self.conv1Modality2.weight=torch.nn.init.kaiming_uniform_(self.conv1Modality2.weight)
        self.conv1Modality2.bias.data.fill_(0.01)
        #self.bn1.weight.data.fill_(1)
        self.bn1Modality2.bias.data.fill_(0.001)
        
        ##Conv layer 2
        self.conv2Modality1 = nn.Conv2d(nFeaturesIntermediateLayers, nFeaturesIntermediateLayers*2, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn2Modality1 = nn.BatchNorm2d(nFeaturesIntermediateLayers*2) 
        self.conv2Modality1.weight=torch.nn.init.kaiming_uniform_(self.conv2Modality1.weight)
        self.conv2Modality1.bias.data.fill_(0.01)
        #self.bn2Modality1.weight.data.fill_(1)
        self.bn2Modality1.bias.data.fill_(0.001)
        
        
        self.conv2Modality2 = nn.Conv2d(nFeaturesIntermediateLayers, nFeaturesIntermediateLayers*2, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn2Modality2 = nn.BatchNorm2d(nFeaturesIntermediateLayers*2) 
        self.conv2Modality2.weight=torch.nn.init.kaiming_uniform_(self.conv2Modality2.weight)
        self.conv2Modality2.bias.data.fill_(0.01)
        #self.bn2Modality1.weight.data.fill_(1)
        self.bn2Modality2.bias.data.fill_(0.001)
        
        ##Conv layer 3
        self.conv3Modality1 = nn.Conv2d(nFeaturesIntermediateLayers*2, nFeaturesIntermediateLayers*2, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn3Modality1 = nn.BatchNorm2d(nFeaturesIntermediateLayers*2)
        self.conv3Modality1.weight=torch.nn.init.kaiming_uniform_(self.conv3Modality1.weight)
        self.conv3Modality1.bias.data.fill_(0.01)
        #self.bn3Modality1.weight.data.fill_(1)
        self.bn3Modality1.bias.data.fill_(0.001)
        
        self.conv3Modality2 = nn.Conv2d(nFeaturesIntermediateLayers*2, nFeaturesIntermediateLayers*2, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn3Modality2 = nn.BatchNorm2d(nFeaturesIntermediateLayers*2)
        self.conv3Modality2.weight=torch.nn.init.kaiming_uniform_(self.conv3Modality2.weight)
        self.conv3Modality2.bias.data.fill_(0.01)
        #self.bn3Modality1.weight.data.fill_(1)
        self.bn3Modality2.bias.data.fill_(0.001)
        
        
        
        ##Conv layer 4
        self.conv4Modality1 = nn.Conv2d(nFeaturesIntermediateLayers*2, nFeaturesIntermediateLayers*2, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn4Modality1 = nn.BatchNorm2d(nFeaturesIntermediateLayers*2)
        self.conv4Modality1.weight=torch.nn.init.kaiming_uniform_(self.conv4Modality1.weight)
        self.conv4Modality1.bias.data.fill_(0.01)
        #self.bn3Modality1.weight.data.fill_(1)
        self.bn4Modality1.bias.data.fill_(0.001)
        
        
        self.conv4Modality2 = nn.Conv2d(nFeaturesIntermediateLayers*2, nFeaturesIntermediateLayers*2, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn4Modality2 = nn.BatchNorm2d(nFeaturesIntermediateLayers*2)
        self.conv4Modality2.weight=torch.nn.init.kaiming_uniform_(self.conv4Modality2.weight)
        self.conv4Modality2.bias.data.fill_(0.01)
        #self.bn4Modality1.weight.data.fill_(1)
        self.bn4Modality2.bias.data.fill_(0.001)
        
        
        ##Conv layer 5
        self.conv5Modality1 = nn.Conv2d(nFeaturesIntermediateLayers*2, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn5Modality1 = nn.BatchNorm2d(nFeaturesIntermediateLayers)
        self.conv5Modality1.weight=torch.nn.init.kaiming_uniform_(self.conv5Modality1.weight)
        self.conv5Modality1.bias.data.fill_(0.01)
        self.bn5Modality1.bias.data.fill_(0.001)
        
        
        self.conv5Modality2 = nn.Conv2d(nFeaturesIntermediateLayers*2, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1, padding=paddingSize ) 
        self.bn5Modality2 = nn.BatchNorm2d(nFeaturesIntermediateLayers)
        self.conv5Modality2.weight=torch.nn.init.kaiming_uniform_(self.conv5Modality2.weight)
        self.conv5Modality2.bias.data.fill_(0.01)
        self.bn5Modality2.bias.data.fill_(0.001)
        
        
         
        ##Conv layer 6
        self.conv6 = nn.Conv2d(nFeaturesIntermediateLayers, nFeaturesFinalLayer, kernel_size=1, stride=1, padding=0 )
        self.bn6 = nn.BatchNorm2d(nFeaturesFinalLayer)
        self.conv6.weight=torch.nn.init.kaiming_uniform_(self.conv6.weight)
        self.conv6.bias.data.fill_(0.01)
        #self.bn6.weight.data.fill_(1)
        self.bn6.bias.data.fill_(0.001)
        

    def forward(self, xModality1, xModality2):
        #x = self.Vgg16features(x)
        xModality1 = self.conv1Modality1(xModality1)
        xModality1 = F.relu( xModality1 )
        xModality1 = self.bn1Modality1(xModality1)
        xModality1 = self.conv2Modality1(xModality1)
        xModality1 = F.relu( xModality1 )
        xModality1 = self.bn2Modality1(xModality1)
        xModality1 = self.conv3Modality1(xModality1)
        xModality1 = F.relu( xModality1 )
        xModality1 = self.bn3Modality1(xModality1)
        xModality1 = self.conv4Modality1(xModality1)
        xModality1 = F.relu( xModality1 )
        xModality1 = self.bn4Modality1(xModality1)
        xModality1 = self.conv5Modality1(xModality1)
        xModality1 = F.relu( xModality1 )
        xModality1 = self.bn5Modality1(xModality1)
        xModality1 = self.conv6(xModality1)
        xModality1 = self.bn6(xModality1)  ##Note there is no relu between last conv and bn
        
        xModality2 = self.conv1Modality2(xModality2)
        xModality2 = F.relu( xModality2 )
        xModality2 = self.bn1Modality2(xModality2)
        xModality2 = self.conv2Modality2(xModality2)
        xModality2 = F.relu( xModality2 )
        xModality2 = self.bn2Modality2(xModality2)
        xModality2 = self.conv3Modality2(xModality2)
        xModality2 = F.relu( xModality2 )
        xModality2 = self.bn3Modality2(xModality2)
        xModality2 = self.conv4Modality2(xModality2)
        xModality2 = F.relu( xModality2 )
        xModality2 = self.bn4Modality2(xModality2)
        xModality2 = self.conv5Modality2(xModality2)
        xModality2 = F.relu( xModality2 )
        xModality2 = self.bn5Modality2(xModality2)
        xModality2 = self.conv6(xModality2)
        xModality2 = self.bn6(xModality2)  ##Note there is no relu between last conv and bn
        
        return xModality1, xModality2
