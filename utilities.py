# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Sudipan Saha
"""


import numpy as np
import scipy.io as sio
import cv2

 


def matchSegmentationResultToOriginalLabel(resultMap, referenceMap):
    
    ##ADDING 1 to not keep any zero value
    ##Otherwise zero is an object here (Impervious surfaces)
    resultMap = resultMap+1
    referenceMap = referenceMap+1
    
    ##Finding unique values
    resultMapUniqueVals = np.unique(resultMap)
    
    referenceMapUniqueVals,referenceMapUniqueCounts = np.unique(referenceMap, return_counts=True)
    referenceSortingIndices = np.argsort(-referenceMapUniqueCounts)
    referenceMapUniqueVals = referenceMapUniqueVals[referenceSortingIndices]
    
    
    resultToReferenceRelationMatrix = np.zeros((len(resultMapUniqueVals),len(referenceMapUniqueVals)))
    
    totalIntersection = 0
    for resultIndex, resultUniqueVal in enumerate(resultMapUniqueVals):
        resultUniqueValIndicator = np.copy(resultMap)
        resultUniqueValIndicator[resultUniqueValIndicator!=resultUniqueVal] = 0
        for referenceIndex,referenceUniqueVal in enumerate(referenceMapUniqueVals):
            referenceUniqueValIndicator = np.copy(referenceMap)
            referenceUniqueValIndicator[referenceUniqueValIndicator!=referenceUniqueVal] = 0
            resultReferenceIntersection = resultUniqueValIndicator*referenceUniqueValIndicator
            numIntersection = len(np.argwhere(resultReferenceIntersection))
            totalIntersection = totalIntersection+numIntersection
            resultToReferenceRelationMatrix[resultIndex,referenceIndex] = numIntersection
            
    resultMapReassigned = np.zeros(resultMap.shape)
    
    for referenceIndex,referenceUniqueVal in enumerate(referenceMapUniqueVals):
        matchesCorrespondingToThisVal = resultToReferenceRelationMatrix[:,referenceIndex]
        if np.sum(matchesCorrespondingToThisVal)==0: ##this check is important, other python finds a max even in a all-zero column
            continue
        maximizingIndex = np.argsort(matchesCorrespondingToThisVal)[-1]
        resultMapOptimumMatch = resultMapUniqueVals[maximizingIndex]
        resultMapReassigned[resultMap==resultMapOptimumMatch] = referenceUniqueVal
        resultToReferenceRelationMatrix[maximizingIndex,:] = 0
        
    ##Subtracting 1 to keep values as it were
    resultMapReassigned = resultMapReassigned-1
    
    ##if some label in the result map has not been assigned to any class of reference, then it will have value -1 at this stage
    ##we reassign it to 6, since "6" indicates undefined class in Potsdam dataset
    resultMapReassigned[resultMapReassigned==-1]=6 
    
    return resultMapReassigned.astype(int)
    
    
    

def matchSegmentationResultToOriginalLabelZurich(resultMap, referenceMap):
    
    
    ##ADDING 1 to not keep any zero value
    ##Otherwise zero is an object here (Impervious surfaces)
    resultMap = resultMap+1
    referenceMap = referenceMap+1
    
    ##Finding unique values
    resultMapUniqueVals = np.unique(resultMap)
    
    referenceMapUniqueVals,referenceMapUniqueCounts = np.unique(referenceMap, return_counts=True)
    referenceSortingIndices = np.argsort(-referenceMapUniqueCounts)
    referenceMapUniqueVals = referenceMapUniqueVals[referenceSortingIndices]
    
    
    resultToReferenceRelationMatrix = np.zeros((len(resultMapUniqueVals),len(referenceMapUniqueVals)))
    
    totalIntersection = 0
    for resultIndex, resultUniqueVal in enumerate(resultMapUniqueVals):
        resultUniqueValIndicator = np.copy(resultMap)
        resultUniqueValIndicator[resultUniqueValIndicator!=resultUniqueVal] = 0
        for referenceIndex,referenceUniqueVal in enumerate(referenceMapUniqueVals):
            referenceUniqueValIndicator = np.copy(referenceMap)
            referenceUniqueValIndicator[referenceUniqueValIndicator!=referenceUniqueVal] = 0
            resultReferenceIntersection = resultUniqueValIndicator*referenceUniqueValIndicator
            numIntersection = len(np.argwhere(resultReferenceIntersection))
            totalIntersection = totalIntersection+numIntersection
            resultToReferenceRelationMatrix[resultIndex,referenceIndex] = numIntersection
            
#    print(resultToReferenceRelationMatrix.astype(int))
    
    resultMapReassigned = np.zeros(resultMap.shape)
    
    for referenceIndex,referenceUniqueVal in enumerate(referenceMapUniqueVals):
        matchesCorrespondingToThisVal = resultToReferenceRelationMatrix[:,referenceIndex]
        if np.sum(matchesCorrespondingToThisVal)==0: ##this check is important, other python finds a max even in a all-zero column
            continue
        maximizingIndex = np.argsort(matchesCorrespondingToThisVal)[-1]
        resultMapOptimumMatch = resultMapUniqueVals[maximizingIndex]
        #print(resultMapOptimumMatch)
        #print(referenceUniqueVal)
        resultMapReassigned[resultMap==resultMapOptimumMatch] = referenceUniqueVal
        resultToReferenceRelationMatrix[maximizingIndex,:] = 0
#    print(np.unique(resultMap))
#    print(np.unique(referenceMap))
      
    ##Subtracting 1 to keep values as it were
    resultMapReassigned = resultMapReassigned-1
    
    ##if some label in the result map has not been assigned to any class of reference, then it will have value -1 at this stage
    ##we reassign it to 8, since "8" indicates uncategorized/background class in Zurich dataset
    resultMapReassigned[resultMapReassigned==-1]=8 
    
    
    return resultMapReassigned.astype(int)
        
        
        




    
    
    
   
             
        




    
    
    
    
    






    



