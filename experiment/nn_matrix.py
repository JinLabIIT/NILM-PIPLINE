import numpy as np

def MAE(y_truth, y_pred):
    return np.abs(y_truth - y_pred).sum()/ y_truth.shape[0] / y_truth.shape[1]

def MAPE(y_truth, y_pred):
    # return np.abs((y_truth - y_pred) / y_truth).sum() / y_truth.shape[0] / y_truth.shape[1]
    # to avoid divide by zero from y_truth
    err = y_truth - y_pred
    y_truth = (y_truth == 0) * err + y_truth
    
    return np.abs(err / y_truth).sum() / y_truth.shape[0] / y_truth.shape[1]
    
    
def NRMSE(y_truth, y_pred):
    y_std = y_truth.std(axis=0)
    return np.sqrt((np.square(y_truth-y_pred) / y_std).sum() / y_truth.shape[0] / y_truth.shape[1])
    
def TOTAL_MAE(y_truth, y_pred):
    return np.abs(y_truth.sum(axis=1) -  y_pred.sum(axis=1)).sum() / y_truth.shape[0]

def TOTAL_MAPE(y_truth, y_pred):
    # return np.abs((y_truth.sum(axis=1) -  y_pred.sum(axis=1))/y_truth.sum(axis=1)).sum() / y_truth.shape[0]
    # to avoid divide by zero from y_truth
    err = y_truth.sum(axis=1) -  y_pred.sum(axis=1)
    y_truth_sum = (y_truth.sum(axis=1) == 0) * err + y_truth.sum(axis=1)
    
    return np.abs(err/y_truth_sum).sum() / y_truth.shape[0]