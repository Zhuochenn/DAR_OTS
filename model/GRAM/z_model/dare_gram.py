import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from mode_2.dare_gram import model
import numpy as np
import os
import configargparse
from mode_2.dare_gram import data_loader_dare as data_loader
import random
from utils import str2bool

def get_parser():

    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    # general configuration
    parser.add("--config", is_config_file=True, default="mode_2/dare_gram/dare.yaml")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # network related
    # parser.add_argument('--backbone', type=str, default='resnet50')

    # data loading related
    parser.add_argument('--src_dir', type=str, default="data/JAN16/source/train")  
    parser.add_argument('--tar_dir', type=str, default="data/JAN16/target/train") 
    parser.add_argument('--fp', type=str, default="forcePosition/force.csv")
    parser.add_argument('--img', type=str, default="images")
    parser.add_argument('--rfimage_source', type=str, default="data/JAN16/source/ref.jpg")
    parser.add_argument('--rfimage_target', type=str, default="data/JAN16/target/ref.jpg")
    parser.add_argument('--mean_std', type=str, default="mean_std.npy")

    # training related
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--test_interval', type=int, default=200)
    parser.add_argument('--num_iter', type=int, default=20000)
    # parser.add_argument('--n_epoch', type=int, default=100)

    # optimizer related
    parser.add_argument('--lr_backbone', type=float, default=1e-1)
    parser.add_argument('--lr_reg', type=float, default=1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    parser.add_argument('--tradeoff_angle', type=float, default=0.05)
    parser.add_argument('--tradeoff_scale', type=float, default=0.001)
    parser.add_argument('--treshold', type=float, default=0.9)
    
    return parser


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
  
    src_img = os.path.join(args.src_dir, args.img)
    src_fp = os.path.join(args.src_dir, args.fp)

    tgt_img = os.path.join(args.tar_dir, args.img)
    tgt_fp = os.path.join(args.tar_dir, args.fp)

    source_data_loader = data_loader.load_data_source(
        src_img, src_fp, args.rfimage_source, args.batch_size, args.mean_std, shuffle=True, num_workers=args.num_workers)
    
    target_data_loader,valid_data_loader = data_loader.load_data_target(
        tgt_img, tgt_fp, args.rfimage_target, args.batch_size, args.mean_std, shuffle=True, num_workers=args.num_workers)
    
    return source_data_loader, target_data_loader, valid_data_loader


# def get_optimizer(model, args):
#     optimizer_dict = [{"params": filter(lambda p: p.requires_grad, model.base_network.parameters()), "lr": args.lr_backbone},
#                     {"params": filter(lambda p: p.requires_grad, model.classifier_layer.parameters()), "lr": args.lr_reg}]
#     optimizer = torch.optim.SGD(optimizer_dict, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
#     return optimizer


def get_optimizer(model, args):
    optimizer_dict = [{"params": filter(lambda p: p.requires_grad, model.base_network.parameters()), "lr": args.lr_backbone},
                    {"params": filter(lambda p: p.requires_grad, model.reg_layer.parameters()), "lr": args.lr_reg}]
    optimizer = torch.optim.SGD(optimizer_dict, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    return optimizer

def Regression_test(loader, model, args):
    MSE = [0, 0, 0, 0]
    MAE = [0, 0, 0, 0]
    number = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            pred = model(imgs)
            MSE[0] += torch.nn.MSELoss(reduction='sum')(pred[:, 0], labels[:, 0])
            MAE[0] += torch.nn.L1Loss(reduction='sum')(pred[:, 0], labels[:, 0])
            MSE[1] += torch.nn.MSELoss(reduction='sum')(pred[:, 1], labels[:, 1])
            MAE[1] += torch.nn.L1Loss(reduction='sum')(pred[:, 1], labels[:, 1])
            MSE[2] += torch.nn.MSELoss(reduction='sum')(pred[:, 2], labels[:, 2])
            MAE[2] += torch.nn.L1Loss(reduction='sum')(pred[:, 2], labels[:, 2])
            MSE[3] += torch.nn.MSELoss(reduction='sum')(pred, labels[:,:3])
            MAE[3] += torch.nn.L1Loss(reduction='sum')(pred, labels[:,:3])
            number += imgs.size(0)
    for j in range(4):
        MSE[j] = MSE[j] / number
        MAE[j] = MAE[j] / number
    print("\tMSE : {0},{1},{2}".format(MSE[0],MSE[1],MSE[2]))
    print("\tMAE : {0},{1},{2}".format(MAE[0], MAE[1], MAE[2]))
    print("\tMSEall : {0}".format(MSE[3]))
    print("\tMAEall : {0}".format(MAE[3]))
    return MAE[3]


def DARE_GRAM_LOSS(H1, H2, args):    
    b,p = H1.shape

    A = torch.cat((torch.ones(b,1).to(args.device), H1), 1)
    B = torch.cat((torch.ones(b,1).to(args.device), H2), 1)

    cov_A = (A.t()@A)
    cov_B = (B.t()@B) 

    _,L_A,_ = torch.linalg.svd(cov_A)
    _,L_B,_ = torch.linalg.svd(cov_B)
    
    eigen_A = torch.cumsum(L_A.detach(), dim=0)/L_A.sum()
    eigen_B = torch.cumsum(L_B.detach(), dim=0)/L_B.sum()

    if(eigen_A[1]>args.treshold):
        T = eigen_A[1].detach()
    else:
        T = args.treshold
        
    index_A = torch.argwhere(eigen_A.detach()<=T)[-1]

    if(eigen_B[1]>args.treshold):
        T = eigen_B[1].detach()
    else:
        T = args.treshold

    index_B = torch.argwhere(eigen_B.detach()<=T)[-1]
    
    k = max(index_A, index_B)[0]

    A = torch.linalg.pinv(cov_A ,rtol = (L_A[k]/L_A[0]).detach())
    B = torch.linalg.pinv(cov_B ,rtol = (L_B[k]/L_B[0]).detach())
    
    cos_sim = nn.CosineSimilarity(dim=0,eps=1e-6)
    cos = torch.dist(torch.ones((p+1)).to(args.device),(cos_sim(A,B)),p=1)/(p+1)
    
    return args.tradeoff_angle*(cos) + args.tradeoff_scale*torch.dist((L_A[:k]),(L_B[:k]))/k

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.1, weight_decay=0.001):
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_lr[i]* (1 + gamma * iter_num) ** (-power)
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer

# class Model_Regression(nn.Module):
#     def __init__(self):
#         super(Model_Regression,self).__init__()
#         self.base_network = model.Resnet18Fc()
#         self.classifier_layer = nn.Linear(512, 3)
#         self.classifier_layer.weight.data.normal_(0, 0.01)
#         self.classifier_layer.bias.data.fill_(0.0)
#         self.classifier_layer = nn.Sequential(self.classifier_layer,  nn.Sigmoid())
#         self.predict_layer = nn.Sequential(self.base_network,self.classifier_layer)

#     def forward(self,x):
#         feature = self.base_network(x)
#         outC= self.classifier_layer(feature)
#         return(outC, feature)

class Model_Regression(nn.Module):
    def __init__(self):
        super(Model_Regression,self).__init__()
        self.base_network = model.Resnet50Fc()
        reg_layer = [
            nn.Linear(self.base_network.output_num(), 256),
            nn.Sigmoid(),
            nn.Linear(256, 3)
        ]
        #final layer
        self.reg_layer = nn.Sequential(*reg_layer)
        self.predict_layer = nn.Sequential(self.base_network,self.reg_layer)

    def forward(self,x):
        feature = self.base_network(x)
        y= self.reg_layer(feature)
        return(y, feature)
    

def main():

    #parse argument
    parser = get_parser()
    args = parser.parse_args()
    set_random_seed(args.seed)
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)

    #Datloader
    source_data_loader, target_data_loader, valid_data_loader = load_data(args)
    len_source = len(source_data_loader) - 1
    len_target = len(target_data_loader) - 1
    iter_source = iter(source_data_loader)
    iter_target = iter(target_data_loader)
    print("len_source: {}, len_source: {}".format(len_source,len_target))
    #Model
    # Model_R = torch.load('mode_2/dare_gram/dare_checkpoint_ref.pth')
    # Model_R.to(args.device)
    Model_R = Model_Regression()
    Model_pretrained = torch.load('mode_2/dare_gram/dare_checkpoint_ref_fewshot.pth')
    state_dict = Model_R.state_dict()
    state_dict.update(Model_pretrained.state_dict())
    Model_R.load_state_dict(state_dict)
    Model_R.to(args.device)
    optimizer = get_optimizer(Model_R, args)
    criterion = {"regressor": nn.MSELoss()}

    param_lr = []

    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])

    train_regression_loss = train_dare_gram_loss = train_total_loss =  0.0

    for iter_num in range(1, args.num_iter + 1):

        Model_R.train(True)
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=args.lr_gamma, power=args.lr_decay, weight_decay=args.weight_decay)
        optimizer.zero_grad()
        if iter_num % len_source == 0:
            iter_source = iter(source_data_loader)
        if iter_num % len_target == 0:
            iter_target = iter(target_data_loader)

        inputs_source, labels_source = next(iter_source)
        inputs_target, labels_target = next(iter_target)
        
        inputs_s, inputs_t = inputs_source.to(args.device), inputs_target.to(args.device)
        labels = labels_source.float().to(args.device)

        outC_s, feature_s = Model_R(inputs_s)
        outC_t, feature_t = Model_R(inputs_t)

        regression_loss = criterion["regressor"](outC_s, labels[:,:3])
        dare_gram_loss = DARE_GRAM_LOSS(feature_s,feature_t, args)

        total_loss = regression_loss + dare_gram_loss
        total_loss.backward()
        optimizer.step()

        train_regression_loss += regression_loss.item()
        train_dare_gram_loss += dare_gram_loss.item()
        train_total_loss += total_loss.item()

        if iter_num % args.test_interval == 0:

            print("Iter: [{:d}/{:d}], reg_loss(MSE): {:.4f}, dare-gram loss: {:.4f}, total_loss: {:.4f}".format(
                iter_num, args.num_iter + 1, train_regression_loss / float(args.test_interval), train_dare_gram_loss / float(args.test_interval), train_total_loss / float(args.test_interval)))
            train_regression_loss = train_dare_gram_loss = train_total_loss =  0.0
            Model_R.eval()
            mae = Regression_test(valid_data_loader, Model_R.predict_layer, args)
            torch.save(Model_R,'mode_2/dare_gram/dare_checkpoint_ref_fewshot.pth.pth')

    # torch.save(Model_R,'dare_gram/dare.pth')

if __name__ == '__main__':
    main()
