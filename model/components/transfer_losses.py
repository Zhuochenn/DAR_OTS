import torch
import torch.nn as nn
from model.components.loss_funcs import *

class TransferLoss(nn.Module):
    def __init__(self, loss_type, **kwargs):
        super(TransferLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == "lmmd":
            self.loss_func = LMMDLoss(**kwargs)
        elif loss_type == "adv":
            self.loss_func = AdversarialLoss(**kwargs)
        elif loss_type == 'gram':
            self.loss_func = GramLoss(**kwargs)
    
    def forward(self, source, target, **kwargs):
        return self.loss_func(source, target, **kwargs)   