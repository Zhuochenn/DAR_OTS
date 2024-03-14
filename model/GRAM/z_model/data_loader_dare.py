import os
import pandas as pd
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.dataset import random_split

class I2FPDataset(Dataset):
    
    def __init__(self, img_path, fp_path, rfimage, mean_std):
        self.img_path = img_path
        self.refimage = rfimage
        self.force_position = pd.read_csv(fp_path)
        self.transform = transforms.Compose(
            [transforms.Resize([256, 256]),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
        mean_std = np.load(mean_std)
        self.force_position['Fx(N)'] = (self.force_position['Fx(N)'] - mean_std[0]) /mean_std[3]
        self.force_position['Fy(N)'] = (self.force_position['Fy(N)'] - mean_std[1]) /mean_std[4]
        self.force_position['Fz(N)'] = (self.force_position['Fz(N)'] - mean_std[2]) /mean_std[5]
    
    def __len__(self):
        return len(self.force_position)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_path, f"{idx}.jpg")  
        image = np.array(Image.open(img_name))
        image_ref = np.array(Image.open(self.refimage))
        image = np.concatenate((image,image_ref),axis=0)
        image = Image.fromarray(image)
        fp = torch.tensor(self.force_position.iloc[idx, :].values, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, fp

def load_data_source(img_path, fp_path,rfimage_source, batch_size, mean_std, shuffle, num_workers=0, **kwargs):
    
    data_set = I2FPDataset(img_path,fp_path,rfimage_source, mean_std)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True,**kwargs)
    
    return data_loader  

def load_data_target(img_path, fp_path, rfimage_target, batch_size, mean_std, shuffle, validation_split=0.2, num_workers=0, **kwargs):
    
    data_set = I2FPDataset(img_path,fp_path, rfimage_target, mean_std)
    valid_size = int(validation_split * len(data_set))
    train_size = len(data_set) - valid_size
    train_dataset, val_dataset  = random_split(data_set,[train_size,valid_size])

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True,**kwargs)
    valid_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True,**kwargs)
    
    return train_data_loader, valid_data_loader

