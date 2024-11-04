import torch
from torch import nn
from utils import *

### loss
def GetLogLossTensor(model, X):
    eps = 1e-20
    p = model(X)
    loss_tensor = -torch.log(p + eps).double()
    loss_tensor = loss_tensor.reshape((loss_tensor.shape[0], loss_tensor.shape[1], 1))
    return loss_tensor

### Models
def Regularizer(model, loss, weight_decay):
    return loss + 0.5 * weight_decay * sum(p.norm(2)**2 for p in model.parameters())

class LogReg(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(input_size, num_classes-1)
        )

    def forward(self, x):
        device = x.device.type
        logits = self.layers(x)
        logits = torch.hstack((torch.ones((logits.shape[0],1)).to(device), logits))
        return nn.functional.softmax(logits, dim=1)

class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size):
        super().__init__()
        # First linear layer (input to hidden)
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        # Second linear layer (hidden to output)
        self.output_layer = nn.Linear(hidden_size, num_classes-1)

    def forward(self, x):
        # Pass input through the hidden layer and apply a non-linear activation
        hidden_output = nn.functional.relu(self.hidden_layer(x))
        # Pass the output of the hidden layer to the output layer
        logits = self.output_layer(hidden_output)
        # Stack with an additional column of ones for the bias term
        device = x.device.type
        logits = torch.hstack((torch.ones((logits.shape[0], 1)).to(device), logits))
        # Apply softmax to get probabilities
        return nn.functional.softmax(logits, dim=1)

    
### Training
def CIRisk(loss_tensor, Z, set_Z, P_Y_Z, device):
    p = OneHotEncode(Z, set_Z, device)@P_Y_Z.T
    return (loss_tensor.squeeze() * p).sum(dim=1).mean()

def TrainModelCI(model, X, Z, set_Z, P_Y_Z, weight_decay, tol, max_epochs, device):
 
    ### Create optimizer ###
    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')

    ### Defining closure ###
    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad() 
        loss_tensor = GetLogLossTensor(model, X)
        loss = CIRisk(loss_tensor, Z, set_Z, P_Y_Z, device)
        loss = Regularizer(model, loss, weight_decay)
        if loss.requires_grad:
            loss.backward()
        return loss
        
    ### Run the training loop ###
    hist = []
    for epoch in range(int(max_epochs)): 
        loss = optimizer.step(closure)
        model_norm = GetGradNorm(model)
        if model_norm<tol:
            break
            
    ### Output ###
    return model


def TrainMLP(model, X, Z, set_Z, P_Y_Z, lr, weight_decay, max_epochs, device):
    # Create optimizer - using Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Run the training loop
    for epoch in range(int(max_epochs)):
        optimizer.zero_grad()  # Clear gradients
        loss_tensor = GetLogLossTensor(model, X)
        loss = CIRisk(loss_tensor, Z, set_Z, P_Y_Z, device)
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters

    # Output
    return model
