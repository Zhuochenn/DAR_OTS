import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import yaml

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
    # os.remove(in_dir)
    fp.to_csv(out_dir, index=False)


def keep_row_test(in_dir,out_dir,rows):

    fp = pd.read_csv(in_dir)
    fp = fp.iloc[rows:,:]
    # os.remove(in_dir)
    fp.to_csv(out_dir, index=False)

def copy_files(file, source_folder, dest_folder):
    shutil.copy(os.path.join(source_folder, file), os.path.join(dest_folder, file))
    

def tt_split():

    base_folder = 'data/JAN16/target/'
    source_images_folder = os.path.join(base_folder, 'images')
    
    train_images_folder = os.path.join(base_folder, 'train/images')
    test_images_folder = os.path.join(base_folder, 'test/images')
    train_forces_folder = os.path.join(base_folder, 'train/forcePosition')
    test_forces_folder = os.path.join(base_folder, 'test/forcePosition')
    
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(test_images_folder, exist_ok=True)
    os.makedirs(train_forces_folder, exist_ok=True)
    os.makedirs(test_forces_folder, exist_ok=True)


    forces_files = 'data/JAN16/target/forcePosition/force.csv'
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

#.../v3/
# create folder and move result
def move_inf(inf_path,folder_name):
    root = os.listdir(inf_path)
    m = ['DANN','DSAN','GRAM']
    # m = ['GRAM'] # source_only
    for i in root:
        if (not i.endswith('txt')) and ('-' in i):
            for j in range(len(m)):
                pth = os.path.join(inf_path,i,m[j]) #.../DANN/
                target_folder = os.path.join(pth,folder_name)
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                [shutil.move(os.path.join(pth,k),target_folder) for k in os.listdir(pth) if os.path.isfile(os.path.join(pth,k))]
    print('successful!')

def modify_yaml(pth,target_dict):
    # m = ['GRAM']
    m = ['DANN','DSAN','GRAM']
    for i in range(len(m)):
        subroot = os.path.join(pth,m[i]) #../DANN
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

def error_table(dir,folder):
    root = os.listdir(dir)
    m = ['DANN','GRAM','DSAN']
    # m = ['GRAM']
    t = {}
    for i in root:
        if (not i.endswith('txt')) and ('-' in i):
            column = []
            for j in range(len(m)):
                pth = os.path.join(dir,i,m[j],folder,"et_s.txt")
                if not os.path.exists(pth):
                    column.append(1000)
                    continue
                e = np.loadtxt(pth,dtype=np.float32,delimiter=',').round(decimals=3)
                column.append(e[2])
            t[i] = column
    table = pd.DataFrame(data=t)
    table = table[['mb0i0-wmb1i1','mb0i0-wmb2i2','wmb1i1-mb0i0','wmb1i1-wmb2i2','inpainting-wmb2i2','inpainting-wmb1i1',\
                   'wmb1i1-inpainting','mb0i0-inpainting','inpainting-mb0i0']]
    print(table)
    table.to_excel('error_table.xlsx')
    print('Done!')



if __name__ == "__main__":

    # ori_path =  '/home/zhuochen/data/Jan16'
    # data_path = 'data/JAN16'
    # st = ['source','target']
    # # create different paths, but source and target have the same datas
    # for i in st:
    #     # data/source (target)/1../
    #     root = os.path.join(data_path,i)
    #     # subroot = os.path.join(root,str(len(os.listdir(root))+1))
    #     #image
    #     # print(os.listdir(ori_path))
    #     img_dir = os.path.join(ori_path,os.listdir(ori_path)[1]) # /home/zhuo/data/data/3/20231220_...
    #     imgnew_dir = root +"/images"    # data/source(target)/1../images
    #     shutil.copytree(img_dir,imgnew_dir)
    #     rename_image(imgnew_dir)
    #     #forces
    #     force_dir = os.path.join(ori_path,os.listdir(ori_path)[0])      
    #     force_new_dir = root + "/forcePosition"   # data/source(target)/1../forcePosition
    #     add_label(force_dir,force_new_dir)
    
    
    # # img_path =  'data/data/source/train/images'
    # # # # del_image(img_path)
    # # rename_image(img_path)

    # # # Fx_max, Fy_max, Fz_max = 0.6,0.6,2.5
    # fp = pd.read_csv('/home/zhuochen/DeepDA/data/wmb1i1/ori/forcePosition/force.csv')
    # max_fx, mean_fx, std_fx = max(abs(i) for i in (fp['Fx(N)'].max(),fp['Fx(N)'].min())),fp['Fx(N)'].mean(),fp['Fx(N)'].std()
    # max_fy, mean_fy, std_fy = max(abs(i) for i in (fp['Fy(N)'].max(),fp['Fy(N)'].min())),fp['Fy(N)'].mean(),fp['Fy(N)'].std()
    # max_fz, mean_fz, std_fz = max(abs(i) for i in (fp['Fz(N)'].max(),fp['Fz(N)'].min())),fp['Fz(N)'].mean(),fp['Fz(N)'].std()

    # max_fx, mean_fx, std_fx = fp['Fx(N)'].min(),fp['Fx(N)'].mean(),fp['Fx(N)'].std()
    # max_fy, mean_fy, std_fy = fp['Fy(N)'].min(),fp['Fy(N)'].mean(),fp['Fy(N)'].std()
    # max_fz, mean_fz, std_fz = fp['Fz(N)'].min(),fp['Fz(N)'].mean(),fp['Fz(N)'].std()

    # max_fx, mean_fx, std_fx = fp['Fx(N)'].max(),fp['Fx(N)'].mean(),fp['Fx(N)'].std()
    # max_fy, mean_fy, std_fy = fp['Fy(N)'].max(),fp['Fy(N)'].mean(),fp['Fy(N)'].std()
    # max_fz, mean_fz, std_fz = fp['Fz(N)'].max(),fp['Fz(N)'].mean(),fp['Fz(N)'].std()
    
    # stat = np.array([mean_fx,mean_fy,mean_fz,std_fx,std_fy,std_fz])
    # np.save('mean_std',stat)
    # print("Fx: max {}, mean {}, std {}".format(max_fx,mean_fx,std_fx))
    # print("Fy: max {}, mean {}, std {}".format(max_fy,mean_fy,std_fy))
    # print("Fz: max {}, mean {}, std {}".format(max_fz,mean_fz,std_fz))

    # tt_split()
    # move_inf('inference/v3','source_only')
    target_dict={"train": False,"use_pretrained":True,'use_checkpoint':True,\
                "draw_tsne":True,"draw_force":False,"transfer_weight":False,\
                "n_epoch": 20, "lr":0.1, "inf_dir":"inference/v3_2", \
                "clf_weight":1,"reg_weight": 0,"transfer_weight": 0, \
                'data_save':'model_trans','num_workers':4,\
                'inf_s': 's.png','inf_t': 't.png','tSNE':'tSNE.png'}
    modify_yaml('model',target_dict)
    # error_table("inference/v3","source_only")
    pass