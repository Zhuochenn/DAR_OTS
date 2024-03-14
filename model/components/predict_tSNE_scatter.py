import os
import torch
import numpy as np
import model.components.data_loader as data_loader
import model.components.models as models
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
import utils.utils as utils
import pandas as pd

def load_data(tar_dir, fp, img, ref_dir, batch_size,mean_std, shuffle,num_workers):

    tgt_img = os.path.join(tar_dir, img)
    tgt_fp = os.path.join(tar_dir, fp)
    target_test_loader = data_loader.load_data_source(tgt_img, tgt_fp, ref_dir, batch_size, mean_std,shuffle,num_workers)
    return target_test_loader

def visual_predict(label, predict,inf_dir):
    plt.figure(figsize=(5,2))
    plt.rcParams["font.size"] = '8'
    ax = plt.subplot(1,2,1)
    ax.scatter(label[:,0],predict[:,0],s=5,alpha = 0.7,color= 'orange',label='Fx')
    ax.scatter(label[:,1],predict[:,1],s=5,alpha = 0.7,color= 'green',label='Fy')
    xreg = np.linspace(-0.75,0.75)
    ax.plot(xreg,xreg,color='darkgrey',lw=1, linestyle = 'dashed')
    r2_x = r2_score(label[:,0],predict[:,0])
    r2_y = r2_score(label[:,1],predict[:,1])
    ax.text(0.3, -0.5, r'$R^2=$' + f'{r2_x:.2f}', fontsize=6,color='orange')
    ax.text(0.3, -0.7, r'$R^2=$' + f'{r2_y:.2f}', fontsize=6,color='green')
    ax.set_ylabel("Prediction(N)")
    ax.set_xlabel("Groundtruth(N)")
    ax.set_xticks(np.arange(-0.6,0.65,0.6))
    ax.set_yticks(np.arange(-0.6,0.65,0.6))
    ax.legend(loc=0)


    ax = plt.subplot(1,2,2)
    ax.scatter(label[:,2],predict[:,2],s=5,alpha = 0.7,color= 'lightcoral',label='Fz')
    xreg = np.linspace(-2.5,0.6)
    ax.plot(xreg,xreg,color='darkgrey',lw=1, linestyle = 'dashed')
    r2_z = r2_score(label[:,2],predict[:,2])
    ax.text(-0.4, -2.2, r'$R^2=$' + f'{r2_z:.2f}', fontsize=6,color='lightcoral')
    ax.set_xticks(np.arange(-2.5,0.6,1))
    ax.set_yticks(np.arange(-2.5,0.6,1))
    ax.set_ylabel("Prediction(N)")
    ax.set_xlabel("Groundtruth(N)")
    ax.legend(loc=0)

    plt.tight_layout() 
    plt.savefig(inf_dir, dpi = 600)  
    plt.close()

def visual_predict_xyz(label, predict,inf_dir):
    plt.figure(figsize=(6,2))
    plt.rcParams["font.size"] = '8'
    # locator= ticker.LinearLocator(numticks=5)
    ax = plt.subplot(1,3,1)
    ax.scatter(label[:,0],predict[:,0],s=20,alpha = 0.7,edgecolor ='turquoise',label='Fx')
    b,a = np.polyfit(label[:,0],predict[:,0],deg=1)
    xreg = np.linspace(np.min(label[:,0]),np.max(label[:,0]))
    ax.plot(xreg,b*xreg+a,color='lightcoral',lw=2)
    ax.set_ylabel("Prediction(N)")
    ax.set_xticks(np.arange(-0.3,0.31,0.2))
    ax.set_yticks(np.arange(-0.3,0.31,0.2))
    # ax.yaxis.set_major_locator(locator)
    # ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    # ax.yaxis.set_major_locator(locator)
    ax.legend(loc=4)

    ax = plt.subplot(1,3,2)
    ax.scatter(label[:,1],predict[:,1],s=20,alpha = 0.7,edgecolor ='turquoise',label='Fy')
    b,a = np.polyfit(label[:,1],predict[:,1],deg=1)
    xreg = np.linspace(np.min(label[:,1]),np.max(label[:,1]))   
    ax.plot(xreg,b*xreg+a,color='lightcoral',lw=2)
    
    ax.set_xticks(np.arange(-0.3,0.31,0.2))
    ax.set_yticks(np.arange(-0.3,0.31,0.2))
    # ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    # ax.yaxis.set_major_locator(locator)
    ax.legend(loc=4)

    ax = plt.subplot(1,3,3)
    ax.scatter(label[:,2],predict[:,2],s=20,alpha = 0.7,edgecolor ='turquoise',label='Fz')
    b,a = np.polyfit(label[:,2],predict[:,2],deg=1)
    xreg = np.linspace(np.min(label[:,2]),np.max(label[:,2]))
    ax.plot(xreg,b*xreg+a,color='lightcoral',lw=2)
    ax.set_xticks(np.arange(-2,0.2,1))
    ax.set_yticks(np.arange(-2,0.2,1))
    # ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    # ax.yaxis.set_major_locator(locator)
    ax.legend(loc=4)

    plt.tight_layout() 
    plt.savefig(inf_dir, dpi = 600)  
    plt.close()

def predict_force(model, inf_dir,mean_std,target_test_loader,tSNE,args):
    if args.train:
        es = 'es.txt'
        et = 'et.txt'
    else:
        es = 'es_s.txt'
        et = 'et_s.txt' 
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean_std = np.load(mean_std)
    # print(mean_std)
    d_label = []
    d_prediction = []
    e_x_r = utils.AverageMeter()                                             
    e_y_r = utils.AverageMeter()                                             
    e_z_r = utils.AverageMeter()                                             
    e_t_r = utils.AverageMeter()  
    criterion_reg = torch.nn.L1Loss()
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(devices), target.to(devices)
            prediction,_ = model.predict(data)
            prediction[:,0] = prediction[:,0]*1.5+(-0.75)
            prediction[:,1] = prediction[:,1]*1.5+(-0.75)
            prediction[:,2] = prediction[:,2]*3+(-3)
            label = target[:,:3].to(torch.float32)
            label[:,0] = label[:,0]*1.5+(-0.75)
            label[:,1] = label[:,1]*1.5+(-0.75)
            label[:,2] = label[:,2]*3+(-3)
            e_x = criterion_reg(prediction[:,0], label[:,0])
            e_y = criterion_reg(prediction[:,1], label[:,1])
            e_z = criterion_reg(prediction[:,2], label[:,2])
            e_t = criterion_reg(prediction, label)
            e_x_r.update(e_x.item())
            e_y_r.update(e_y.item())
            e_z_r.update(e_z.item())
            e_t_r.update(e_t.item())
            d_label.append(label.cpu().data.numpy())
            d_prediction.append(prediction.cpu().data.numpy())
            if not tSNE and len(d_label) == len(target_test_loader):
                visual_predict(np.array(d_label).reshape(-1,3),np.array(d_prediction).reshape(-1,3),inf_dir)
        if not tSNE:
            error = [e_x_r.avg,e_y_r.avg,e_z_r.avg,e_t_r.avg]
            root = os.path.dirname(inf_dir)
            if 't' not in os.path.split(inf_dir)[1]:
                np.savetxt(os.path.join(root,es),error,delimiter=',')
                print('Source: ex {:.4f} ey {:.4f} ez {:.4f} et {:.4f}'.format(e_x_r.avg,e_y_r.avg,e_z_r.avg,e_t_r.avg))
            else:
                np.savetxt(os.path.join(root,et),error,delimiter=',')
                print('Target: ex {:.4f} ey {:.4f} ez {:.4f} et {:.4f}'.format(e_x_r.avg,e_y_r.avg,e_z_r.avg,e_t_r.avg))

def inf_force(args, checkpoint_dir,tar_dir,ref_dir,inf_dir,shuffle=False,batch_size=32,num_workers=1,tSNE=False,tSNE_pth='source'):
    
    features_in_hook = []

    def hook(module, fea_in, fea_out):
        features_in_hook.append(fea_in[0].data.cpu().numpy().flatten())
        return None
        
    model = torch.load(checkpoint_dir)
    model.eval()
    model.reg_layer.register_forward_hook(hook=hook)
    target_test_loader = load_data(tar_dir, args.fp, args.img, ref_dir, batch_size, args.mean_std,shuffle, num_workers)
    predict_force(model,inf_dir,args.mean_std,target_test_loader, tSNE,args)
    if tSNE:
        np.save(tSNE_pth,features_in_hook)



def tSNE_gen_old(source, target,tSNE_dir,args):
    source = np.load(source)
    target = np.load(target)
    X_ = np.concatenate((source, target))
    X_tsne = TSNE(2).fit_transform(X_)  
    plt.figure(figsize=(4, 4))
    plt.plot(X_tsne[:len(source), 0], X_tsne[:len(source), 1], '.', label="source",markersize=1,color='lightcoral')
    plt.plot(X_tsne[len(source):, 0], X_tsne[len(source):, 1], 'x', label="target",markersize=1,color='turquoise')
    plt.axis('off')
    plt.savefig(tSNE_dir,dpi=600)
    plt.close()

def tSNE_gen(source, target,tSNE_dir,args):
    s_label = pd.read_csv(os.path.join(args.tSNE_source_dir,args.fp))
    t_label = pd.read_csv(os.path.join(args.tSNE_target_dir,args.fp))
    s_label = s_label.iloc[:,3]
    t_label = t_label.iloc[:,3]
    selected = [0,40,80,120,160,200,240,280]
    source = np.load(source)
    target = np.load(target)
    source_selected = []
    s_label_selected = []
    target_selected = []
    t_label_selected = []

    for i in range(len(source)):
        if s_label[i] in selected:
            source_selected.append(source[i])
            s_label_selected.append(s_label.iat[i])
    
    for j in range(len(target)):
        if t_label[j] in selected:
            target_selected.append(target[j])
            t_label_selected.append(t_label.iat[j])

    X_ = np.concatenate((source_selected, target_selected))
    X_tsne = TSNE(2,perplexity=100).fit_transform(X_)
    source_selected = X_tsne[:len(source_selected),:]
    target_selected = X_tsne[len(source_selected):,:]
    color = {0:'blue',40:'orange',80:'green',120:'red',160:'purple',200:'brown',240:'pink',280:'cyan'}
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    for i in range(len(source_selected)):
        plt.plot(source_selected[i][0], source_selected[i][1], marker = '.', label="source",markersize=6,color=color[s_label_selected[i]])
    for j in range(len(target_selected)):
        plt.plot(target_selected[i][0], target_selected[i][1], marker = 's', label="target",markersize=6,color=color[t_label_selected[j]])
    plt.savefig(tSNE_dir,dpi=600)
    plt.close()