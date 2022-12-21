from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from lightning_model import KelpClassifier
from torch.utils.data import Subset
import torchvision
import torch
import random as rnd
import numpy as np
from torch.utils.data import DataLoader, random_split
import os

print("Loading the model...")
model = KelpClassifier.load_from_checkpoint(

    "/pvol/logs/lightning_logs/version_27/checkpoints/epoch=99-step=22200.ckpt",
    #arch="inception_v3",
    pretrained=True,
    optimizer="Adam",
    criterion="cross_entropy",
    #lr=0.00005,
    #data_classes=data_classes,
)

img_size = 512
batch_size = 32

print("Loading dataset loader...")
test_set = torchvision.datasets.ImageFolder('/pvol' + '/' + str(img_size)+ '_images/', transform= transforms.ToTensor()) 

idx = []
counter = [0,0]
for i in range(len(test_set)):
#   mask[i] = True
    if test_set.imgs[i][1] == test_set.class_to_idx['Ecklonia']:
        if rnd.randint(0, 50) == 1 : # here you can define percentage of Ecklonia entries
            
            counter[0] += 1 
            idx.append(i)
    else: 
        idx.append(i)
        counter[1] += 1 

subset = Subset(test_set, idx)
print('Ecklonia counter: {} \nOthers counter : {} \nPercentage Ecklonia: {}'.format(counter[0], counter[1], counter[0]/len(subset)))
print('The subset length is {} of {}'.format(len(subset), (counter[0]+counter[1])))


test_loader =  torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

from pytorch_lightning import Trainer
# Instantiate the Trainer
trainer = Trainer(
    num_nodes=1,
    accelerator='gpu',
    devices=1,
    default_root_dir='/pvol/test/'

)
# Start testing
trainer.test(model, test_loader)