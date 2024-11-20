import torch 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
import sys

def GetGradNorm(model):
    return torch.sqrt(sum(p.grad.norm(2)**2 for p in model.parameters() if p.requires_grad))

def DefineDevice(device):
    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    else:
        return device
        
def OneHotEncode(Y, set_Y, device=None):
    one_hot = torch.zeros(Y.shape[0], set_Y.shape[0]).to(device)
    one_hot[torch.arange(Y.shape[0]), Y.long()] = 1
    return one_hot.double()

## Specifically for the experiments ##
def FindRowIndex(X, x):
    x = x.view(1, -1)
    for i in range(X.shape[0]):
        if torch.all(torch.eq(X[i], x)):
            return i 
    return -1

def GetP_Y_Z(Y, Z, set_Y, set_Z):
   
    P_Y_Z = torch.zeros((set_Z.shape[0], set_Y.shape[0]))
    for z in set_Z:
        ind = Z == z
        if ind.float().sum()==0:
            p=1/len(set_Y)
            for y in range(len(set_Y)):
                P_Y_Z[z][y] = p
        else:
            for y in range(len(set_Y)):
                P_Y_Z[z][y] = (Y[ind]==y).float().mean().item()    

    P_Y_Z = P_Y_Z.T
    return P_Y_Z

class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        
""" In case the practitioner wants to learn the dependence graph
https://github.com/JieyuZ2/wrench/blob/main/wrench/labelmodel/dependency_structure.py
https://github.com/HazyResearch/metal/blob/cb_deps/tutorials/Learned_Deps.ipynb


import cvxpy as cp
import numpy as np
import scipy as sp


def get_deps_from_inverse_sig(J, thresh=0.2):
    deps = []
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            if abs(J[i, j]) > thresh:
                deps.append((i, j))
    return deps


def learn_structure(L, thresh=1.5):
    N = float(np.shape(L)[0])
    M = L.shape[1]
    sigma_O = (np.dot(L.T, L)) / (N - 1) - \
              np.outer(np.mean(L, axis=0), np.mean(L, axis=0))

    # bad code
    O = 1 / 2 * (sigma_O + sigma_O.T)
    O_root = np.real(sp.linalg.sqrtm(O))

    # low-rank matrix
    L_cvx = cp.Variable([M, M], PSD=True)

    # sparse matrix
    S = cp.Variable([M, M], PSD=True)

    # S-L matrix
    R = cp.Variable([M, M], PSD=True)

    # reg params
    lam = 1 / np.sqrt(M)
    gamma = 1e-8

    objective = cp.Minimize(
        0.5 * (cp.norm(R @ O_root, 'fro') ** 2) - cp.trace(R) + lam * (gamma * cp.pnorm(S, 1) + cp.norm(L_cvx, "nuc")))
    constraints = [R == S - L_cvx, L_cvx >> 0]

    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=False)
    opt_error = prob.value

    # extract dependencies
    J_hat = S.value

    deps_hat = get_deps_from_inverse_sig(J_hat, thresh=thresh)
    deps = [(i, j) for i, j in deps_hat if i < j]
    return deps

"""