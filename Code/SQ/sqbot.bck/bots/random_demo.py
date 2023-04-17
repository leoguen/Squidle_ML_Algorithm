import random
from sqapi.annotate import Annotator, register_annotator_plugin
import pickle
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
#from PyTorch/lightning_model.py import KelpClassifier
#from PyTorch/lightning_model.py import MixedDataset
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
import pandas as pd
from pytorch_lightning import Trainer
import sys

sys.path.insert(0,'/home/leo/Documents/IMAS/Code/PyTorch')
from lightning_model import KelpClassifier


class SQDataset(Dataset):
    def __init__(self, img):
        self.data = [img]

        
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, idx):
        img = self.data[idx]
        img_tensor = transforms.ToTensor()(img)
        class_id = 1
        return img_tensor, class_id
    


class RandoBOT(Annotator):
#class RandoBOT(): #for Testing locally
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.possible_codes = ["ECK", "ASC", "SUB"]
        print("Loading the model...")
        self.model = KelpClassifier.load_from_checkpoint(
            "/home/leo/Documents/IMAS/Code/SQ/sqbot/models/epoch=12-step=13689.ckpt",
            pretrained=True,
            optimizer="Adam",
            criterion="cross_entropy",
            backbone_name = 'regnet_x_32gf',
            no_filters = 0
        )

        self.img_size = 288
        self.batch_size = 1
        self.test_perc=0.001
        
    """
    def classify_point(self, mediaobj, x, y, t):

        # image_data = mediaobj.data()            # cv2 image object containing media data
        # media_path = mediaobj.url               # path to media item
        print(f"CLASSIFYING: {mediaobj.url} | x: {x},  y: {y},  t: {t}")
        classifier_code = random.sample(self.possible_codes, 1)[0]
        prob = round(random.random(), 2)
        return classifier_code, prob
    """
        
    def classify_point(self, mediaobj, x, y, t):
        """
        #Overridden method: predict label for x-y point
        """
        BOUNDING_BOX_SIZE = [self.img_size,self.img_size]
        image_data = mediaobj.data()            # cv2 image 
        #image_data = cv2.imread('/home/leo/Documents/IMAS/Code/SQ/sqbot/bots/Ecklonia_radiata_578887.jpg') # for testing locally
        # object containing media data
        x = image_data.shape[1]*x
        y = image_data.shape[0]*y
        # crop around coordinates
        cropped_image = image_data[
            int(y-(BOUNDING_BOX_SIZE[0]/2)):int(y+(BOUNDING_BOX_SIZE[0]/2)), 
            int(x-(BOUNDING_BOX_SIZE[1]/2)):int(x+(BOUNDING_BOX_SIZE[1]/2))]
        #print(f"CLASSIFYING: {mediaobj.url} | x: {x},  y: {y},  t: {t}")
        print('CLASSIFYING')
        test_set = SQDataset(cropped_image)
        test_loader =  torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count())

        # Instantiate the Trainer
        trainer = Trainer(
            num_nodes=1,
            accelerator='cpu',
            devices=1,
        )

        results = (trainer.predict(self.model, test_loader))
        #id, y, top_class, top_p in results
        if results[0][2] == 0: classifier_code = 'ECK'
        else: classifier_code = 'OTHERS'  
        prob = results[0][3]


        
        #classifier_code = random.sample(self.possible_codes, 1)[0]
        #prob = round(random.random(), 2)
        
        return classifier_code, prob
''' For testing locally
if __name__ == "__main__":
    code = RandoBOT()
    mediaobj = 0
    x = 0.5
    y = 0.5
    t = 1
    code.classify_point(mediaobj, x, y, t)
'''

register_annotator_plugin(RandoBOT, name="RandoBOT")