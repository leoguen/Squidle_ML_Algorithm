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

CKPT_PATH = '/home/ubuntu/IMAS/Code/SQ/sqbot/models/epoch=92-step=122295.ckpt'
CNN_DIR_PATH = '/home/ubuntu/IMAS/Code/PyTorch'

sys.path.insert(0,CNN_DIR_PATH)

from lightning_model import KelpClassifier

class SQDataset(Dataset):

    def __init__(self, img):
        self.data = [img]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        train_transforms = transforms.Compose([
            #transforms.Resize((299, 299)),
        transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])

        img_tensor = train_transforms(img)
        class_id = 1
        return img_tensor, class_id

class RegnetBOT(Annotator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.possible_codes = ["ECK", "ASC", "SUB"]
        print("Loading the model...")

        self.model = KelpClassifier.load_from_checkpoint(
        CKPT_PATH,
        pretrained=True,
        optimizer="Adam",
        criterion="cross_entropy",
        backbone_name = 'regnet_x_32gf',
        no_filters = 0,
        )

        acc_val = 'cpu'

        if torch.cuda.is_available(): acc_val = 'gpu'
        # Instantiate the Trainer
        self.trainer = Trainer(
        num_nodes=1,
        accelerator=acc_val,
        devices=1,
        logger = False,
        #no_filters = 0,
        num_sanity_val_steps=0,
        precision=16,
        )
        self.img_size = 288
        self.batch_size = 1
        self.test_perc=0.001


    def get_crop_points(self, x, y, original_image, BOUNDING_BOX_SIZE):

        crop_dist = BOUNDING_BOX_SIZE[1]/2
        if x - crop_dist < 0: x0 = 0
        else: x0 = x - crop_dist

        if y - crop_dist < 0: y0 = 0
        else: y0 = y - crop_dist

        if x + crop_dist > original_image.shape[1]: x1 = original_image.shape[1]
        else: x1 = x + crop_dist

        if y + crop_dist > original_image.shape[0]: y1 = original_image.shape[0]
        else: y1 = y + crop_dist

        return int(x0), int(x1), int(y0), int(y1)



    def classify_point(self, mediaobj, x, y, t):

        """
        #Overridden method: predict label for x-y point
        """

        BOUNDING_BOX_SIZE = [self.img_size,self.img_size]
        image_data = mediaobj.data() # cv2 image
        x = image_data.shape[1]*x
        y = image_data.shape[0]*y
        # crop around coordinates

        x0, x1, y0, y1 = self.get_crop_points(x, y, image_data, BOUNDING_BOX_SIZE)

        cropped_image = image_data[y0:y1, x0:x1]

        test_set = SQDataset(cropped_image)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count())

        results = (self.trainer.predict(self.model, test_loader))
        #id, y, top_class, top_p in results

        if results[0][2] == 0:
            classifier_code = 'ECK'
            prob = results[0][3]
            print(classifier_code, prob)
            return classifier_code, prob
        else:
            print('NO_ECK', 0)
            return 'ECK', 0

register_annotator_plugin(RegnetBOT, name="RegnetBOT")