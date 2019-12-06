import numpy as np
import os, sys


def getConfusionMatrix(YTrue, YPredict):
    """
    Computes the confusion matrix.
    Parameters
    ----------
    YTrue : numpy array
        This array contains the ground truth.
    YPredict : numpy array
        This array contains the predictions.
    Returns
    CM : numpy matrix
        The confusion matrix.
    """
    len_labels = len(np.unique(YTrue))
    cm = np.zeros((len_labels ,len_labels ), int )
    for i in range(len(YTrue)):
        cm[int(YTrue[i])][int(YPredict[i])] = cm[int(YTrue[i])][int(YPredict[i])] + 1
    return cm
                
    
def getPerformanceScores(YTrue, YPredict):
    """
    Computes the accuracy, precision, recall, f1 score.
    Parameters
    ----------
    YTrue : numpy array
        This array contains the ground truth.
    YPredict : numpy array
        This array contains the predictions.
    Returns
    {"CM" : numpy matrix,
    "accuracy" : float,
    "precision" : float,
    "recall" : float,
    "f1" : float}
        This should be a dictionary.
    """
    cm = getConfusionMatrix(YTrue,YPredict)

    true_values = np.sum(np.diagonal(cm))
    accuracy = true_values/len(YTrue)

    precision = np.zeros(cm.shape[0])
    recall = np.zeros(cm.shape[0])

    for i in range(cm.shape[0]):
        print(cm[i][i],cm[0:,i])
        precision[i] = cm[i][i]/np.sum(cm[0:,i])
    for i in range(cm.shape[0]):
        recall[i] = cm[i][i]/np.sum(cm[i])
    f1 = np.divide(2*(np.multiply(precision,recall)),precision+recall)

    result =  {"CM" : cm, "accuracy" : accuracy, "precision" : precision, "recall" : recall, "f1" : f1}
    return result