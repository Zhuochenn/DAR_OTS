import os
import torch
import numpy as np
import model.components.data_loader2 as data_loader
import model.components.models as models
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import utils.utils as utils

def load_data(tar_dir, fp, img, ref_dir, batch_size,mean_std, shuffle,num_workers):

    tgt_img = os.path.join(tar_dir, img)
    tgt_fp = os.path.join(tar_dir, fp)
    target_test_loader = data_loader.load_data_source(tgt_img, tgt_fp, ref_dir, batch_size, mean_std,shuffle,num_workers)
    return target_test_loader

def visual_predict(label, predict,inf_dir):
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(label[:,0],label = "groundtruth",color = "lightcoral")
    plt.plot(predict[:,0],label = "prediction",color = "turquoise")
    plt.title("Force Prediction")
    plt.ylabel("Fx/N")

    plt.subplot(3,1,2)
    plt.plot(label[:,1],label = "groundtruth",color = "lightcoral")
    plt.plot(predict[:,1],label = "prediction",color = "turquoise")
    plt.ylabel("Fy/N")

    plt.subplot(3,1,3)
    plt.plot(label[:,2],label = "groundtruth",color = "lightcoral")
    plt.plot(predict[:,2],label = "prediction",color = "turquoise")
    plt.ylabel("Fz/N")
    plt.xlabel("Point")

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
            prediction[:,0] = prediction[:,0]*mean_std[3]+mean_std[0]
            prediction[:,1] = prediction[:,1]*mean_std[4]+mean_std[1]
            prediction[:,2] = prediction[:,2]*mean_std[5]+mean_std[2]
            label = target[:,:3].to(torch.float32)
            label[:,0] = label[:,0]*mean_std[3]+mean_std[0]
            label[:,1] = label[:,1]*mean_std[4]+mean_std[1]
            label[:,2] = label[:,2]*mean_std[5]+mean_std[2]
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
            if not tSNE and len(d_label) == 64:
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



def tSNE_gen(source, target,tSNE_dir):
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

