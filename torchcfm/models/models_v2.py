import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self,w):
        super(ResBlock,self).__init__()
        self.linear = nn.Linear(w,w)
        self.selu = nn.SELU()

    def forward(self,x):
        return self.selu(self.linear(self.selu(self.linear(x))) + x) 
    
class ResNet(nn.Module):
    def __init__(self,dim,out_dim = None, w = 64, hidden = 2,time_varying = False):
        super(ResNet,self).__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        # The * unpacks the list into positional arguments to be evaluated
        self.layers = nn.ModuleList()
        # Initial layer allows to include time
        self.layers.append(nn.Linear(dim + (1 if time_varying else 0), w)),
        self.layers.append(nn.SELU())
        # Assemble ResNet architecture
        for _ in range(hidden):
            self.layers.append(ResBlock(w))
        # Final layer
        self.layers.append(nn.Linear(w,out_dim))
        
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GradModel(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, x):
        x = x.requires_grad_(True)
        grad = torch.autograd.grad(torch.sum(self.action(x)), x, create_graph=True)[0]
        return grad[:, :-1]
