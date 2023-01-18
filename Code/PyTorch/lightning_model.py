from argparse import ArgumentParser
import os
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.models as models
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import loggers as pl_loggers
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


class KelpClassifier(pl.LightningModule):
    def __init__(self, backbone_name, no_filters, learning_rate):
        super().__init__()
        # init a pretrained resnet       
        backbone = getattr(models, backbone_name)(weights='DEFAULT')
        #backbone = models.regnet_x_32gf(weights ='DEFAULT')
        #num_filters = 2520
        num_filters = no_filters
        if num_filters == 0:
            num_filters = backbone.fc.in_features
        
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        num_target_classes = 2
        
        self.classifier = nn.Linear(num_filters,  num_target_classes)
        self.save_hyperparameters()

        '''
        self.l1 = torch.nn.Linear(img_size * img_size * 3, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 2)
        '''

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        
        #x = x.view(x.size(0), -1)
        #x = torch.relu(self.l1(x))
        #x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        batch_idx = image_to_tb(self, batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
    
    def predict_step(self, batch, batch_idx):
        # This can be used for implementation
        x, y = batch
        y_hat = self(x)
        #loss = F.cross_entropy(y_hat, y)
        prob = F.softmax(y_hat, dim=1)
        top_p, top_class = prob.topk(1, dim = 1)
        
        return batch_idx, int(y[0]), int(top_class.data[0][0]), float(top_p.data[0][0])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

def image_to_tb(self, batch, batch_idx):
    
    if batch_idx == 0:
        tensorboard = self.logger.experiment
        others_batch_images = []
        ecklonia_batch_images = []
        for index, image in enumerate(batch[0]):
            if batch[1][index] == 1: others_batch_images.append(image) 
            else: ecklonia_batch_images.append(image)
        others_grid = torchvision.utils.make_grid(others_batch_images)
        ecklonia_grid = torchvision.utils.make_grid(ecklonia_batch_images)
        tensorboard.add_image('Ecklonia images', ecklonia_grid, batch_idx)
        tensorboard.add_image('Other images', others_grid, batch_idx)
        tensorboard.close()
        #tensorboard.add_image(batch[0])
    return batch_idx

def cli_main():
    #writer = SummaryWriter(log_dir='/pvol/runs/')
    
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    batch_size = 32
    learning_rate = 0.0001

    # ------------
    # data
    # ------------

    # Create datasets for training & validation, download if necessary
    full_set = UniformDataset(img_size)
    test_perc = 0.2
    training_set, validation_set = torch.utils.data.random_split(full_set,[0.85, 0.15], generator=torch.Generator().manual_seed(42))
    test_set = MixedDataset(img_size, test_perc)
    
    # Create data loaders for our datasets; shuffle for training and for validation
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    print('Dataset image size: {}'.format(img_size))
    print('Number of training images: {}'.format(len(train_loader.dataset)))
    
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
    
    print('Number of validation images: {}'.format(len(val_loader.dataset)))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=os.cpu_count())
    print('Number of test images: {}'.format(len(test_loader.dataset)))
    
    # ------------
    # model
    # ------------
    model = KelpClassifier(backbone_name, no_filters,learning_rate)
    # ------------
    # training
    # ------------
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="/pvol/logs/lightning_logs", name=str(img_size)+'_'+backbone_name+'_Image_Size')
    if torch.cuda.is_available(): 
        max_epochs= 50
        trainer = pl.Trainer(
                accelerator='gpu', 
                devices=1, 
                max_epochs=max_epochs, 
                logger=tb_logger, 
                default_root_dir='/pvol/',
                auto_lr_find=True, 
                auto_scale_batch_size=True, 
                log_every_n_steps=max_epochs)
    else:
        trainer = pl.Trainer(accelerator='cpu', 
                devices=1, 
                max_epochs=max_epochs, 
                logger=tb_logger, 
                default_root_dir='/pvol/',
                auto_lr_find=True, 
                auto_scale_batch_size=True, 
                log_every_n_steps=max_epochs)
    trainer.fit(model, train_loader, val_loader)
    
    # ------------
    # testing
    # ------------
    trainer.test(ckpt_path='best', dataloaders=test_loader)
    return model 

if __name__ == '__main__':
    #img_list = [16, 24, 32, 64, 128, 224, 240, 256, 272, 288, 304, 320, 336, 512]
    img_list = [300]

    model_specs = [
        ['resnet50', 0],
        #['googlenet', 0], 
        #['convnext_large', 1536], 
        #['convnext_small', 768], 
        ['resnext101_64x4d', 0], 
        ['efficientnet_v2_l', 1280], 
        #['vit_h_14', 1280], 
        ['regnet_x_32gf', 0], 
        #['swin_v2_b', 1024]
        ]

    for img_size in img_list:
        for backbone_name, no_filters in model_specs:
            model = cli_main()
            try:
                cli_main()
            except:
                print('!!! Error \n   Problem with img_size {} and backbone {}'.format(img_size, backbone_name))
                continue
