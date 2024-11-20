import torch
import numpy as np
try:
    from .bound_expectation import BoundExpectation
    from .utils import *
except ImportError:
    from bound_expectation import BoundExpectation
    from utils import *

def ztn(x): #zero to nan
    if x==0:return np.nan
    else: return x
def truncate(value):
    if value < 0:
        return 0
    elif value > 1:
        return 1
    else:
        return value
  
def GetAccuracyTensor(y_hat, set_Y):
    tensor = OneHotEncode(y_hat, set_Y).double()
    tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], 1))
    return tensor

def GetPRFTensor(y_hat, set_Y):
    assert set_Y.tolist()==[0,1], "Function designed for binary classification (set_Y=[0,1])."
    tensor = OneHotEncode(y_hat, set_Y).double()
    tensor[:, 0] = 0
    tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], 1))
    return tensor

def EvalPWS(y_hat, metric,
            set_Y, set_Z, Z_test, P_Y_Z, device, y_hat_full=None, Z_full=None,
            conf=.95, approx_error = .01, tol=1e-4, max_epochs=1e3):

    assert metric in ['accuracy','prf']
    epsilon=approx_error/np.log(set_Y.shape[0])
    
    if metric == 'accuracy':
        tensor = GetAccuracyTensor(y_hat, set_Y).to(device)
    else:
        tensor = GetPRFTensor(y_hat, set_Y).to(device)
        
    bounds = {}
    for bound in ['lower','upper']:
        
        bb = BoundExpectation(bound, tensor, Z_test, set_Z, P_Y_Z,
                              conf=conf, epsilon=epsilon,
                              tol=tol, max_epochs=max_epochs, device=device)
        
        bounds[bound] = {}
        bounds[bound]['center'] = bb[0]
        bounds[bound]['ci'] = bb[1]
        

    if metric == 'accuracy':
        output = bounds
        return output
    
    else:        
        denominator = {}
        denominator['precision'] = ztn(y_hat_full.float().mean().item())
        denominator['recall'] = ztn((OneHotEncode(Z_full, set_Z).double().to(device)@P_Y_Z.T)[:,1].mean().item())
        denominator['f1'] = ztn((y_hat_full.float().mean().item()+OneHotEncode(Z_full, set_Z).double().to(device)@P_Y_Z.T)[:,1].mean().item()/2)
        
        output = {}
        for bound in ['lower','upper']:
            output[bound] = {}
            
            for met in ['precision', 'recall', 'f1']:
                output[bound][met] = {}
                output[bound][met]['center'] = bounds[bound]['center']/denominator[met]
                delta = output[bound][met]['center'] - bounds[bound]['ci'][0]/denominator[met]
                output[bound][met]['ci'] = [output[bound][met]['center']-delta, output[bound][met]['center']+delta]
                   
             
                output[bound][met]['center'] = truncate(output[bound][met]['center'])
                output[bound][met]['ci'][0] = truncate(output[bound][met]['ci'][0])
                output[bound][met]['ci'][1] = truncate(output[bound][met]['ci'][1])
        return output
