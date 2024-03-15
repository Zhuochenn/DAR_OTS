import torch
import torch.nn as nn
import model.components.backbones as backbones
from model.components.transfer_losses import TransferLoss

class TransferNet(nn.Module):
    def __init__(self, out_class, out_dim, base_net='resnet50', transfer_loss='mmd', bottleneck_width=256, max_iter=1000, **kwargs):
        super(TransferNet, self).__init__()
        self.out_class = out_class
        self.out_dim = out_dim
        self.transfer_loss = transfer_loss
        #Resnet50
        self.base_network = backbones.get_backbone(base_net)
        bottleneck = [nn.Linear(self.base_network.output_num(), bottleneck_width),
                      nn.ReLU()]
        self.bottleneck_layer = nn.Sequential(*bottleneck)
        #reg_layer
        reg_layer = [
            nn.Linear(bottleneck_width,3),
            nn.Sigmoid()
        ]
        self.reg_layer = nn.Sequential(*reg_layer)
        self.criterion_reg = torch.nn.MSELoss()
        self.reg_mae = torch.nn.L1Loss()
        #clf_layer
        self.clf_layer = nn.Linear(bottleneck_width,self.out_class)
        self.criterion_clf = torch.nn.CrossEntropyLoss()
        #trans_loss
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class":  self.out_class
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args) 

    def forward(self, source, target, source_fp_label):
        #get label
        clf = source_fp_label[:,-1].to(torch.int64)
        reg = source_fp_label[:,:3].to(torch.float32)
        #forward to bottleneck
        source = self.base_network(source)
        target = self.base_network(target)
        source = self.bottleneck_layer(source)
        target = self.bottleneck_layer(target)
        #clf 
        source_clf = self.clf_layer(source)
        target_clf = self.clf_layer(target)
        #loss_clf
        clf_loss = self.criterion_clf(source_clf, clf)
        #reg
        source_reg = self.reg_layer(source)
        reg_loss = self.criterion_reg(source_reg,reg)
        reg_mae = self.reg_mae(source_reg,reg)
        # loss_transfer
        kwargs = {}
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = clf
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        transfer_loss = self.adapt_loss(source, target, **kwargs)

        return clf_loss, reg_loss, transfer_loss, reg_mae
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr},
            {'params': self.clf_layer.parameters(), 'lr': 1.0 * initial_lr},
            {'params': self.reg_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )      
        return params

    def predict(self, x):
        features = self.base_network(x)
        x = self.bottleneck_layer(features)
        reg = self.reg_layer(x)
        clf = self.clf_layer(x)
        return reg, clf