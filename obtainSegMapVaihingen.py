# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Sudipan Saha
"""
import os
import glob
import sys
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import filters
from skimage import morphology
from skimage.filters import rank
import cv2 

import tifffile 


from skimage.transform import resize
from skimage import filters
from skimage.morphology import disk
from tifffile import imsave

from utilities import matchSegmentationResultToOriginalLabel
from sklearn.metrics import confusion_matrix


nanVar=float('nan')
eps = 1e-14

##Defining parameters
numberOfImageChannels = 3


##Defining data paths
#testSetAreaIDs= [11, 15, 28, 30, 34]
testSetAreaIDs= [11, 15, 28, 30, 34]
noclutter=True  ##ignore clutter in computing accuracy


useCuda = torch.cuda.is_available()   
  
  
##setting manual seeds
manualSeed=40
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
np.random.seed(manualSeed)


    

nanVar=float('nan')


palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter/background (red)
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
        #print(c)
        #print(i)
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d
    
def eval_image(gt, pred, acc1, acc2, acc3, acc4, acc5, noclutter=True):

    im_row, im_col = np.shape(pred)
    cal_classes = 5 if noclutter else 6 # no. of classes to calculate scores

    if noclutter:
        gt[gt == 5] = 6 # pixels in clutter are not considered (regarding them as boundary)

    pred[gt == 6] = 6 # pixels on the boundary are not considered for calculating scores
    OA = np.float32(len(np.where((np.float32(pred) - np.float32(gt)) == 0)[0])-len(np.where(gt==6)[0]))/np.float32(im_col*im_row-len(np.where(gt==6)[0]))
    acc1 = acc1 + len(np.where((np.float32(pred) - np.float32(gt)) == 0)[0])-len(np.where(gt==6)[0])
    acc2 = acc2 + im_col*im_row-len(np.where(gt==6)[0])
    pred1 = np.reshape(pred, (-1, 1))
    gt1 = np.reshape(gt, (-1, 1))
    idx = np.where(gt1==6)[0]
    pred1 = np.delete(pred1, idx)
    gt1 = np.delete(gt1, idx)
    CM = confusion_matrix(pred1, gt1)
    for i in range(cal_classes):
        tp = np.float32(CM[i, i])
        acc3[i] = acc3[i] + tp
        fp = np.sum(CM[:, i])-tp
        acc4[i] = acc4[i] + fp
        fn = np.sum(CM[i, :])-tp
        acc5[i] = acc5[i] + fn
        P = tp/(tp+fp+eps)
        R = tp/(tp+fn+eps)
        f1 = 2*(P*R)/(P+R+eps)

    return acc1, acc2, acc3, acc4, acc5
    

def bgr2index(gt_bgr, eroded=False):
    # mapping BGR W x H x 3 image to W x H x C class index
    # opencv read image to BGR format
    im_col, im_row, _ = np.shape(gt_bgr)
    gt = np.zeros((im_col, im_row, 6)) if not eroded else np.zeros((im_col, im_row, 7))
    gt[(gt_bgr[:, :, 2] == 255) & (gt_bgr[:, :, 1] == 255) & (gt_bgr[:, :, 0] == 255), 0] = 1
    gt[(gt_bgr[:, :, 2] == 0) & (gt_bgr[:, :, 1] == 0) & (gt_bgr[:, :, 0] == 255), 1] = 1
    gt[(gt_bgr[:, :, 2] == 0) & (gt_bgr[:, :, 1] == 255) & (gt_bgr[:, :, 0] == 255), 2] = 1
    gt[(gt_bgr[:, :, 2] == 0) & (gt_bgr[:, :, 1] == 255) & (gt_bgr[:, :, 0] == 0), 3] = 1
    gt[(gt_bgr[:, :, 2] == 255) & (gt_bgr[:, :, 1] == 255) & (gt_bgr[:, :, 0] == 0), 4] = 1
    gt[(gt_bgr[:, :, 2] == 255) & (gt_bgr[:, :, 1] == 0) & (gt_bgr[:, :, 0] == 0), 5] = 1
    if eroded:
        gt[(gt_bgr[:, :, 2] == 0) & (gt_bgr[:, :, 1] == 0) & (gt_bgr[:, :, 0] == 0), 6] = 1

    
    return gt
    
    
def bgr2indexForPred(gt_bgr, eroded=False):
    # mapping BGR W x H x 3 image to W x H x C class index
    # opencv read image to BGR format
    im_col, im_row, _ = np.shape(gt_bgr)
    gt = np.zeros((im_col, im_row, 6)) if not eroded else np.zeros((im_col, im_row, 7))
    gt[(gt_bgr[:, :, 2] == 255) & (gt_bgr[:, :, 1] == 255) & (gt_bgr[:, :, 0] == 255), 0] = 1
    gt[(gt_bgr[:, :, 2] == 0) & (gt_bgr[:, :, 1] == 0) & (gt_bgr[:, :, 0] == 255), 1] = 1
    gt[(gt_bgr[:, :, 2] == 0) & (gt_bgr[:, :, 1] == 255) & (gt_bgr[:, :, 0] == 255), 2] = 1
    gt[(gt_bgr[:, :, 2] == 0) & (gt_bgr[:, :, 1] == 255) & (gt_bgr[:, :, 0] == 0), 3] = 1
    gt[(gt_bgr[:, :, 2] == 255) & (gt_bgr[:, :, 1] == 255) & (gt_bgr[:, :, 0] == 0), 4] = 1
    gt[(gt_bgr[:, :, 2] == 255) & (gt_bgr[:, :, 1] == 0) & (gt_bgr[:, :, 0] == 0), 5] = 1
    if eroded:
        gt[(gt_bgr[:, :, 2] == 0) & (gt_bgr[:, :, 1] == 0) & (gt_bgr[:, :, 0] == 0), 6] = 1
    else:
        gt[(gt_bgr[:, :, 2] == 0) & (gt_bgr[:, :, 1] == 0) & (gt_bgr[:, :, 0] == 0), 5] = 1
    
    return gt
    

for areaID in testSetAreaIDs:
  #print('Processing area ID: '+str(areaID))
  labelPath = '../data/Vaihingen/gt/mask_top_mosaic_09cm_area'+str(areaID)+'.tif'
  imagePath = '../data/Vaihingen/img/top_mosaic_09cm_area'+str(areaID)+'.tif'
  loadModelPath = './trainedModels/top_mosaic_09cm_area'+str(1)+'.pth'
  
  ### Paramsters related to the CNN model
  
  
  
  
  
  
  
  
  
  inputImage = tifffile.imread(imagePath)
  ##inputImage = inputImage[:,:,0:3] ###taking only RGB
  labelImage = tifffile.imread(labelPath)
  labelImagePalette = convert_from_color(labelImage)
  
  ##Since images are in 0-255 range, dividing by 255
  inputImage = inputImage/255
  #dsmImage = dsmImage/255
  
  baseImageFileName = (os.path.basename(imagePath).rsplit(".",1))[0]
  
  
  
  if os.path.isfile(loadModelPath):
          model = torch.load(loadModelPath)
  else:
          sys.exit('Model path not found')   
      
  if useCuda:
          model.cuda()
  
  ##Shape of input image
  inputImageShape = inputImage.shape
  inputImageShapeRow = inputImageShape[0]
  inputImageShapeCol = inputImageShape[1]
  
  segmentationMap1 = np.zeros((inputImageShapeRow,inputImageShapeCol))
  
  
  segmentationStride = 800  ##since cannot process the entire image at a time
  segmentationOverlap = 100  ## intentional overlap, since cannot process the entire image at a time
  for imageRowIter in range(0,inputImageShapeRow,segmentationStride):  ##since Potsdam images are of size 6000*6000
    for imageColIter in range(0,inputImageShapeCol,segmentationStride):  ##since Potsdam images are of size 6000*6000
      
      if imageRowIter==0 and imageColIter==0:
          startingRowIndex = 0
          endingRowIndex = segmentationStride+segmentationOverlap
          startingColIndex = 0
          endingColIndex = segmentationStride+segmentationOverlap
      elif imageRowIter==0:
          startingRowIndex = 0
          endingRowIndex = segmentationStride+segmentationOverlap
          startingColIndex = imageColIter-segmentationOverlap
          endingColIndex = min(imageColIter+segmentationStride+segmentationOverlap,inputImageShapeCol)
      elif imageColIter==0:
          startingRowIndex = imageRowIter-segmentationOverlap
          endingRowIndex = min(imageRowIter+segmentationStride+segmentationOverlap,inputImageShapeRow)
          startingColIndex = 0
          endingColIndex = segmentationStride+segmentationOverlap
      else:
          startingRowIndex = imageRowIter-segmentationOverlap
          endingRowIndex = min(imageRowIter+segmentationStride+segmentationOverlap,inputImageShapeRow)
          startingColIndex = imageColIter-segmentationOverlap
          endingColIndex = min(imageColIter+segmentationStride+segmentationOverlap,inputImageShapeCol)
      
      data1 = inputImage[startingRowIndex:endingRowIndex,startingColIndex:endingColIndex,:]
      #print(data1.shape) 
  
      
      patchToProcessData1= np.copy(data1)
      inputToNetData1=torch.from_numpy(patchToProcessData1).type(torch.cuda.FloatTensor)
      inputToNetData1 = inputToNetData1.permute(2,0,1)
      inputToNetData1 = torch.unsqueeze(inputToNetData1,0)
      
  
      if useCuda:
          inputToNetData1 = inputToNetData1.cuda()    
          
          
      ##Obtaining projection
      model.eval()
      model.requires_grad=False
      with torch.no_grad():
          projection1,_ = model(inputToNetData1,inputToNetData1) 
              
     
      
      _, prediction1 = torch.max(projection1,1)  
      
      ##Obtaining segmentation maps
      prediction1Squeezed = torch.squeeze(prediction1).cpu().numpy().astype(int)
      
      prediction1Squeezed = filters.rank.modal(prediction1Squeezed,disk(3))
      
      if imageRowIter==0 and imageColIter==0: 
          segmentationMap1[0:segmentationStride,0:segmentationStride] = prediction1Squeezed[0:segmentationStride,0:segmentationStride]
      elif imageRowIter==0:
          segmentationMap1[0:segmentationStride,imageColIter:min(imageColIter+segmentationStride,inputImageShapeCol)] =\
                                                             prediction1Squeezed[0:segmentationStride,\
                                                                                segmentationOverlap:min(segmentationOverlap+segmentationStride,prediction1Squeezed.shape[1])]   
      elif imageColIter==0:
          segmentationMap1[imageRowIter:min(imageRowIter+segmentationStride,inputImageShapeRow),imageColIter:min(imageColIter+segmentationStride,inputImageShapeCol)] =\
                                                             prediction1Squeezed[segmentationOverlap:min(segmentationOverlap+segmentationStride,prediction1Squeezed.shape[0]),\
                                                                                  0:segmentationStride] 
      else:
          segmentationMap1[imageRowIter:min(imageRowIter+segmentationStride,inputImageShapeRow),imageColIter:min(imageColIter+segmentationStride,inputImageShapeCol)] =\
                                                              prediction1Squeezed[segmentationOverlap:min(segmentationOverlap+segmentationStride,prediction1Squeezed.shape[0]),\
                                                                                   segmentationOverlap:min(segmentationOverlap+segmentationStride,prediction1Squeezed.shape[1])]   
    
  
  selectedSegmentationMap1 = segmentationMap1 
  
  
  
  selectedSegmentationMap1Remapped = matchSegmentationResultToOriginalLabel(selectedSegmentationMap1.astype(int),labelImagePalette)
  
  selectedSegmentationMap1RemappedColor = convert_to_color(selectedSegmentationMap1Remapped)
  
      
  
  
  plt.imsave('./results/vaihingen/'+baseImageFileName+'.png',selectedSegmentationMap1RemappedColor) 
      
   
             
  
##Compute accuracy      
nb_classes = 5 if noclutter else 6
acc1 = 0.0 # accumulator for correctly classified pixels
acc2 = 0.0 # accumulator for all valid pixels (not including label 0 and 6)
acc3 = np.zeros((nb_classes, 1)) # accumulator for true positives
acc4 = np.zeros((nb_classes, 1)) # accumulator for false positives
acc5 = np.zeros((nb_classes, 1)) # accumulator for false negatives

for areaId in range(len(testSetAreaIDs)):
    imagePath = '../data/Vaihingen/img/top_mosaic_09cm_area'+str(areaID)+'.tif'
    labelPath = '../data/Vaihingen/gt/mask_top_mosaic_09cm_area'+str(areaID)+'.tif' 
    gt = bgr2index(cv2.imread(labelPath))
    gt = np.argmax(gt, -1)
   
    
    baseImageFileName = (os.path.basename(imagePath).rsplit(".",1))[0]

    # predict one image
    pred = bgr2indexForPred(cv2.imread('./results/vaihingen/'+baseImageFileName+'.png'))
    pred = np.argmax(pred, -1)

 


    # evaluate one image
    acc1, acc2, acc3, acc4, acc5 = eval_image(gt, pred, acc1, acc2, acc3, acc4, acc5, noclutter)

OA = acc1/acc2

f1 = np.zeros((nb_classes, 1));
iou = np.zeros((nb_classes, 1));
#ca = np.zeros((nb_classes, 1));
for i in range(nb_classes):
    P = acc3[i]/(acc3[i]+acc4[i])
    R = acc3[i]/(acc3[i]+acc5[i])
    f1[i] = 2*(P*R)/(P+R)
    iou[i] = acc3[i]/(acc3[i]+acc4[i]+acc5[i])
    #ca[i] =  acc3[i]/(acc3[i]+acc4[i])

f1_mean = np.mean(f1)
iou_mean = np.mean(iou)
#ca_mean = np.mean(ca)
print('mean f1:', f1_mean, '\nmean iou:', iou_mean, '\nOA:', OA)


    
    
    
    
    






    



