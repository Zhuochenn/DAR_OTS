import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import yaml
from PIL import Image

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')
    
def rename_image(in_dir):
     
    ori_images = os.listdir(in_dir)
    ori_images.sort(key=lambda x : int(os.path.splitext(x)[0]))
    for index, old_name in enumerate(ori_images):
        new_name = f"{index}.jpg" 
        old_path = os.path.join(in_dir,old_name)
        new_path = os.path.join(in_dir,new_name)
        os.rename(old_path,new_path)
        print(f"Renamed: {old_name} -> {new_name}")

def del_image(in_dir):
     
    ori_images = os.listdir(in_dir)
    ori_images.sort()

    for file_name in ori_images:
        try:
            index = int(os.path.splitext(file_name)[0])
            if index < 7220:
                file_path = os.path.join(in_dir, file_name)
                os.remove(file_path)
                print(f"Deleted: {file_name}")
        except ValueError:
            pass

def add_label(in_dir,out_dir):

    fp = pd.read_csv(in_dir)
    drops = ['Tx(Nmm)','Ty(Nmm)','Tz(Nmm)','Num','TimeStamp']
    fp.drop(columns=drops,inplace=True)
    index_name = 'label'
    fp[index_name] = (fp.index % 361)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    fp.to_csv(out_dir+'/force.csv', index=False)

def keep_row_train(in_dir,out_dir,rows):

    fp = pd.read_csv(in_dir)
    fp = fp.iloc[rows:,:]
    fp.to_csv(out_dir, index=False)


def keep_row_test(in_dir,out_dir,rows):

    fp = pd.read_csv(in_dir)
    fp = fp.iloc[rows:,:]
    fp.to_csv(out_dir, index=False)

def copy_files(file, source_folder, dest_folder):
    shutil.copy(os.path.join(source_folder, file), os.path.join(dest_folder, file))
    

def tt_split(base_folder,forces_files):

    source_images_folder = os.path.join(base_folder, 'images')
    
    train_images_folder = os.path.join(base_folder, 'train/images')
    test_images_folder = os.path.join(base_folder, 'test/images')
    train_forces_folder = os.path.join(base_folder, 'train/forcePosition')
    test_forces_folder = os.path.join(base_folder, 'test/forcePosition')
    
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(test_images_folder, exist_ok=True)
    os.makedirs(train_forces_folder, exist_ok=True)
    os.makedirs(test_forces_folder, exist_ok=True)

    df = pd.read_csv(forces_files)
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)
    train_df.columns = ["Fx(N)","Fy(N)","Fz(N)"]			
    test_df.columns = ["Fx(N)","Fy(N)","Fz(N)"]	
    train_df.to_csv(train_forces_folder+'/force.csv', index=False)
    test_df.to_csv(test_forces_folder+'/force.csv', index=False)
    print(train_df[:10])
    print(test_df[:10])
    ori_images = os.listdir(source_images_folder)
    ori_images.sort(key=lambda x : int(os.path.splitext(x)[0]))
    train_images=[ori_images[i] for i in train_df.index]
    test_images=[ori_images[j] for j in test_df.index]
    print(train_images[:10])
    print(test_images[:10])
    train_count = 20000
    for i in train_images:
        copy_files(i, source_images_folder, train_images_folder)
        new_name = f"{train_count}.jpg" 
        os.rename(os.path.join(train_images_folder,i),os.path.join(train_images_folder,new_name))
        train_count += 1

    test_count = 20000
    for j in test_images:
        copy_files(j, source_images_folder, test_images_folder)
        new_name = f"{test_count}.jpg" 
        os.rename(os.path.join(test_images_folder,j),os.path.join(test_images_folder,new_name))
        test_count += 1
        
    rename_image(train_images_folder)
    rename_image(test_images_folder)
    print("done!")

def modify_yaml(pth,target_dict):
    m = ['DANN','DSAN','GRAM']
    for i in range(len(m)):
        subroot = os.path.join(pth,m[i]) 
        ms = os.listdir(subroot) 
        for domain in ms:
            if '-' in domain:
                dir_yaml = os.path.join(subroot,domain)
                print(f"{dir_yaml}")
                pth_yaml = os.path.join(dir_yaml,[j for j in os.listdir(dir_yaml) if "_so" not in j][0])
                with open(pth_yaml,'r') as f:
                    data = yaml.safe_load(f)
                    for k,v in target_dict.items():
                        data[k] = v
                with open(pth_yaml,'w') as f:
                    yaml.dump(data,f,sort_keys=False)
    print('Done')        


if __name__ == "__main__":
    pass