import os
import torch
import numpy as np
import mode_2.dare_gram.data_loader_dare as data_loader
from mode_2.dare_gram.dare_gram import Model_Regression
import matplotlib 
matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt

img = "images"
fp = "forcePosition/force.csv"
mean_std = './mean_std.npy'
ref_dir = "data/JAN16/source/ref.jpg"
features_in_hook = []

def load_data(tar_dir, fp, img, batch_size,mean_std, shuffle,num_workers):

    tgt_img = os.path.join(tar_dir, img)
    tgt_fp = os.path.join(tar_dir, fp)
    target_test_loader = data_loader.load_data_source(tgt_img, tgt_fp, ref_dir, batch_size, mean_std,shuffle,num_workers)
    return target_test_loader

def visual_predict(label, predict,inf_dir):
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(label[:,0],label = "groundtruth",color = "r")
    plt.plot(predict[:,0],label = "predict",color = "b")
    plt.title("Force (Ground Truth - Predict)")
    plt.ylabel("Fx/N")

    plt.subplot(3,1,2)
    plt.plot(label[:,1],label = "groundtruth",color = "r")
    plt.plot(predict[:,1],label = "predict",color = "b")
    plt.ylabel("Fy/N")

    plt.subplot(3,1,3)
    plt.plot(label[:,2],label = "groundtruth",color = "r")
    plt.plot(predict[:,2],label = "predict",color = "b")
    plt.ylabel("Fz/N")
    plt.xlabel("Point")

    plt.legend(loc='lower right')
    plt.tight_layout() 
    plt.savefig(inf_dir, dpi = 600)  
    plt.show()
    plt.close()

def predict_force(model, inf_dir,mean_std,target_test_loader,tSNE=False):

    mean_std = np.load(mean_std)
    print(mean_std)
    for data, target in target_test_loader:
        data, target = data.to(devices), target.to(devices)
        s_output,_ = model(data)
        s_output = s_output.cpu().data.numpy()
        s_output[:,0] = s_output[:,0]*mean_std[3]+mean_std[0]
        s_output[:,1] = s_output[:,1]*mean_std[4]+mean_std[1]
        s_output[:,2] = s_output[:,2]*mean_std[5]+mean_std[2]
        target_fp = np.array(target[:,:3].cpu().data.numpy())
        target_fp[:,0] = target_fp[:,0]*mean_std[3]+mean_std[0]
        target_fp[:,1] = target_fp[:,1]*mean_std[4]+mean_std[1]
        target_fp[:,2] = target_fp[:,2]*mean_std[5]+mean_std[2]
        if not tSNE:
            visual_predict(target_fp,s_output,inf_dir)

def hook(module, fea_in, fea_out):
    features_in_hook.append(fea_in[0].data.cpu().numpy().flatten())
    return None

if __name__ == "__main__":

    # variables
    load_dir = 'mode_2/dare_gram/dare_checkpoint_ref.pth'
    # tar_dir = "data/JAN16/source/test"
    tar_dir = "data/JAN16/target/test"
    inf_dir = "/home/zhuochen/Desktop/temp/fig/Jan17/dare_t.png" #inference dir
    shuffle = False
    batch_size = 32
    num_workers = 1
    # load model
    model = torch.load(load_dir)
    model.eval()
    model.base_network.register_forward_hook(hook=hook)
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_test_loader = load_data(tar_dir, fp, img, batch_size, mean_std,shuffle, num_workers)
    tSNE = False  # False: only predict and output figure; True: only tSNE, no figure
    # predict force
    predict_force(model,inf_dir,mean_std,target_test_loader,tSNE)
    # tSNE_gen
    if tSNE:
        tSNE_path = os.path.join("DANN/tSNE",tar_dir.split('/')[2])
        np.save(tSNE_path,features_in_hook)
        print("successuful!")

