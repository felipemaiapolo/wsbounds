import torch 
import numpy as np
from scipy.stats import norm
try: from .utils import GetGradNorm, OneHotEncode
except ImportError: from utils import GetGradNorm, OneHotEncode

def BoundExpectation(bound, tensor, Z, set_Z, P_Y_Z, conf, epsilon, tol, max_epochs, device, verbose=False):
    
    assert bound in ['upper', 'lower']
    
    ### Optimizer ###
    lamb = torch.nn.Parameter(torch.zeros((1, P_Y_Z.shape[0], P_Y_Z.shape[1]), dtype=torch.float64, device=device))
    optimizer = torch.optim.LBFGS([lamb], lr=1, line_search_fn='strong_wolfe')
    OneHotZ = OneHotEncode(Z, set_Z, device)
    approx = epsilon*torch.log(torch.tensor(tensor.shape[1]))
    hist = []
    
    ###
    if verbose: print('Approximation:', approx.item(),"\n")
    
    ### Defining closure ###
    output = {}
    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad() 
        
        penalty = ((lamb.mean(dim=1))**2).mean()
        if bound == 'upper':
            individual_eval = (epsilon*torch.logsumexp((tensor + lamb)/epsilon, dim=1) * OneHotZ).sum(axis=1) - (OneHotZ @ (P_Y_Z.T * lamb.squeeze().T)).sum(axis=1)
        else:
            individual_eval = -((-epsilon*torch.logsumexp(-(tensor + lamb)/epsilon, dim=1) * OneHotZ).sum(axis=1) - (OneHotZ @ (P_Y_Z.T * lamb.squeeze().T)).sum(axis=1))

        ### Computing loss
        assert len(individual_eval.shape)==1
        loss = individual_eval.mean() + penalty

        ### Backward pass
        if loss.requires_grad:
            loss.backward()
            
        ### Storing
        delta = (norm.ppf((conf+1)/2)*individual_eval.std()/np.sqrt(individual_eval.shape[0])).item()
        output['loss'] = loss.item()
        output['ci'] = [output['loss']-delta, output['loss']+delta]

        return loss
    
    ### Run the optimization loop ###
    for epoch in range(int(max_epochs)):
        loss = optimizer.step(closure)
        hist.append(output['loss'])
        lamb_norm = lamb.grad.norm(2).item()
        
        if epoch>0:
            if verbose: print('Δf={:.10f} ||∇f||={:.10f}'.format(hist[-2] - hist[-1], lamb_norm))
        
        if epoch>0:
            if hist[-2] - hist[-1] < tol:
                break
            
    if bound=='upper':
        return output['loss'], output['ci']
    else:
        output['loss'] = -output['loss']
        output['ci'] = [-output['ci'][1], -output['ci'][0]]
        return output['loss'], output['ci']
