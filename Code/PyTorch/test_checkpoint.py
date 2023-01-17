from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from lightning_model import KelpClassifier
from torch.utils.data import Subset
import torch
import random as rnd
import numpy as np
from torch.utils.data import DataLoader, random_split
import os
from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
import glob
import cv2
import random

class UniformDataset(Dataset):
    def __init__(self, img_size):
        self.imgs_path = '/pvol' + '/' + str(img_size)+ '_images/'
        file_list = [self.imgs_path + 'Others', self.imgs_path + 'Ecklonia']
        #print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
        #print(self.data)
        self.class_map = {"Ecklonia" : 0, "Others": 1}
        #self.img_dim = (img_size, img_size)
        
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        class_id = self.class_map[class_name]
        img_tensor = transforms.ToTensor()(img)
        class_id = torch.tensor(class_id)
        return img_tensor, class_id

class MixedDataset(Dataset):
    def __init__(self, img_size, test_perc):
        self.data = []
        self.imgs_path = '/pvol' + '/' + str(img_size)+ '_images/'
        self.pad_imgs_path = '/pvol' + '/' + str(img_size)+ '_images/Padding/'
        #####################
        # Get unpadded images
        #####################
        file_list = [self.imgs_path + 'Others', self.imgs_path + 'Ecklonia']
        if test_perc*0.04*2*100 < 1:eck_perc = 1
        else: eck_perc = test_perc*0.04*2*100
        unpad_perc = [test_perc*100, eck_perc]
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                if class_name == 'Others': i = 0
                else: i = 1
                if random.randint(0,99) < int(unpad_perc[i]):
                    self.data.append([img_path, class_name])
        oth_count, eck_count = 0, 0
        for entry in self.data:
            oth_count += entry.count('Others')
            eck_count += entry.count('Ecklonia')
        print('Others: {}, Ecklonia: {}'.format(oth_count, eck_count))
        #####################
        # Get padded images
        #####################
        file_list = [self.pad_imgs_path + 'Others', self.pad_imgs_path + 'Ecklonia']

        oth_pad_files = (len([name for name in os.listdir(file_list[0]) if os.path.isfile(os.path.join(file_list[0], name))])) 
        eck_pad_files =(len([name for name in os.listdir(file_list[1]) if os.path.isfile(os.path.join(file_list[1], name))]))

        len_both = (oth_pad_files+eck_pad_files)
        adap_len_both = len_both*test_perc
        oth_perc = int(test_perc*100)
        if int(test_perc*0.04*100) == 0: eck_perc =1
        else: eck_perc = int(test_perc*0.04*100)
        #eck_perc = int(((adap_len_both)/(2*eck_pad_files))*100)
        #oth_perc = int((adap_len_both/(2*oth_pad_files))*100)
        pad_oth_count = 0
        pad_eck_count = 0
        perc = [oth_perc, eck_perc]
        for idx, class_path in enumerate(file_list):
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                if random.randint(0,99) < perc[idx]:
                    if class_name == 'Others': pad_oth_count +=1
                    else: pad_eck_count +=1 
                    self.data.append([img_path, class_name])
        self.class_map = {"Ecklonia" : 0, "Others": 1}
        print('The dataset comprises of: \nUniform Ecklonia {}\nUniform Others {} \nPadded Ecklonia {}\nPadded Others {}\nDataset length {}'.format(eck_count, oth_count, pad_eck_count, pad_oth_count, len(self.data) ))
    
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        class_id = self.class_map[class_name]
        img_tensor = transforms.ToTensor()(img)
        class_id = torch.tensor(class_id)
        return img_tensor, class_id

def analyze_results(results, thresh):
    true_pos, true_neg, false_pos, false_neg, unlabeled, eck_count, oth_count = 0,0,0,0,0,0,0 
    for y, top_class, top_p in results:
        # Count Eck and Oth
        if y == 0: eck_count +=1
        else: oth_count +=1
        # Check True Positives etc
        if top_p <= thresh:
            unlabeled += 1
        else:
            if y == top_class:
                if y == 0: true_pos += 1
                elif y == 1: true_neg += 1
            elif y != top_class:
                if y == 1: false_pos += 1
                elif y == 0: false_neg += 1
    return true_pos, true_neg, false_pos, false_neg, unlabeled,eck_count, oth_count
print("Loading the model...")
model = KelpClassifier.load_from_checkpoint(

    "/pvol/logs/lightning_logs/300_regnet_x_32gf_Image_Size/version_0/checkpoints/epoch=49-step=107200.ckpt",
    #arch="inception_v3",
    pretrained=True,
    optimizer="Adam",
    criterion="cross_entropy",
    backbone_name = 'regnet_x_32gf',
    no_filters = 0
    #lr=0.00005,
    #data_classes=data_classes,
)

img_size = 300
batch_size = 1

print("Loading dataset loader...")
test_set = MixedDataset(img_size, test_perc=0.3)
#test_set = torchvision.datasets.ImageFolder('/pvol' + '/' + str(img_size)+ '_images/', transform= transforms.ToTensor()) 

test_loader =  torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

from pytorch_lightning import Trainer
# Instantiate the Trainer
trainer = Trainer(
    num_nodes=1,
    accelerator='gpu',
    devices=1,
    default_root_dir='/pvol/test/'
)
results=[]

results.append(trainer.predict(model, test_loader))
thresh_list = [0.5,0.6,0.7,0.8,0.9]
for thresh in thresh_list:
    true_pos, true_neg, false_pos, false_neg, unlabeled,eck_count, oth_count = analyze_results(results[0], thresh)
    accuracy = (true_pos+true_neg)/(true_pos+true_neg+false_neg+false_pos)
    print('True Positives: {}\nTrue Negatives: {}\nFalse Positives: {}\nFalse Negatives: {}\nThreshold: {}\nUnlabeled Ecklonia: {}\nUnlabeled Others: {}\nAccuracy with Threshold: {}'.format(true_pos, true_neg, false_pos, false_neg, thresh,eck_count-true_pos-false_neg, oth_count-true_neg-false_pos, accuracy))
