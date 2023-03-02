from argparse import ArgumentParser
import os
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torchmetrics
import torchvision
import torchvision.models as models
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data.dataset import Dataset
import glob
import cv2
import random
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics.classification import BinaryF1Score
from pytorch_lightning.callbacks import TQDMProgressBar
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryRecall
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
import warnings
import numpy as np
from PIL import Image
import shutil
import argparse
import re
import pandas as pd
from os.path import isfile, join
from os import listdir

writer = SummaryWriter()

class CSV_Dataset(Dataset):
    def __init__(self, img_size, test_list, test, inception, test_img_path):
        self.inception = inception
        self.csv_file_df = pd.read_csv('/pvol/Ecklonia_Database/Original_images/106704_normalized_deployment_key_list.csv')
        
        
        compare_dir_csv(self)
        # Add unpadded and padded entries to data

        '''
        self.imgs_path = IMG_PATH + str(img_size)+ '_images/'
        file_list = [self.imgs_path + 'Others', self.imgs_path + 'Padding/Others', self.imgs_path + 'Ecklonia', self.imgs_path + 'Padding/Ecklonia']
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])

            self.class_map = {"Ecklonia" : 1, "Others": 0}
        '''
    def __len__(self):
        #print(self.csv_file_df.shape[0])
        return self.csv_file_df.shape[0]    
    
    def __getitem__(self, idx):
        """
        if '.jpg/' in self.csv_file_df.point_media_path_best[idx]:
            second_part = str(re.sub(".*/(.*).jpg/", "\\1", self.csv_file_df.point_media_path_best[idx])) 
        elif '.jpg' in self.csv_file_df.point_media_path_best[idx]:
            second_part = str(re.sub(".*/(.*).jpg", "\\1", self.csv_file_df.point_media_path_best[idx]))
        elif '.JPG/' in self.csv_file_df.point_media_path_best[idx]:
            second_part = str(re.sub(".*/(.*).JPG/", "\\1", self.csv_file_df.point_media_path_best[idx])) 
        elif '.JPG' in self.csv_file_df.point_media_path_best[idx]:
            second_part = str(re.sub(".*/(.*).JPG", "\\1", self.csv_file_df.point_media_path_best[idx])) 
        else:
            second_part =str(re.sub(".*/(.*)", "\\1", self.csv_file_df.point_media_path_best[idx]))

        file_name = str(re.sub("\W", "_", self.csv_file_df.point_media_deployment_campaign_key[idx])) +"-"+ re.sub("\W", "_",second_part)
        """
        img_path = '/pvol/Ecklonia_Database/Original_images/All_Images/' + self.csv_file_df.file_name.iloc[idx]
        class_id = torch.tensor(0)
        if self.csv_file_df.label_name.iloc[idx] == 'Ecklonia radiata': 
            class_id = torch.tensor(1)
        img = cv2.imread(img_path)
        #class_id = self.class_map[class_name]
        img = Image.fromarray(img)
        x = self.csv_file_df.point_x.iloc[idx] # At this point only float
        y = self.csv_file_df.point_y.iloc[idx] # At this point only float
        #print(img.size)
        x_img, y_img = img.size
        x = x_img*x #Center position
        y = y_img*y #Center position
        if x > (img_size/2): x = int (x - (img_size/2))
        else: x =0
        y = int (y + (img_size/2))
        if y > y_img: y = y_img
        
        cropped_img = torchvision.transforms.functional.crop(img, y, x, img_size, img_size)
        if self.inception:
            train_transforms = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAutocontrast(p=0.5),
            transforms.RandomEqualize(p=0.4),
            transforms.ColorJitter(brightness=0.5, hue=0.2),
            transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            
            ])
        else:
            train_transforms = transforms.Compose([
            transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])
        img_tensor = train_transforms(cropped_img)
        #print(type(img_tensor))
        #class_id = torch.tensor(class_id)
        return img_tensor, class_id

class GeneralDataset(Dataset):
    def __init__(self, img_size, test_list, test, inception, test_img_path):
        self.inception = inception
        if test: # True test dataset is returned
            if test_list != []: # Dataset should be extracted out of training dataset
                # Add unpadded and padded entries to data
                self.data = test_list[0]
                self.data = self.data + test_list[1]
                self.class_map = {"Ecklonia" : 1, "Others": 0}
            else: # Testdataset should be used
                self.imgs_path = test_img_path + str(img_size)+ '_images/'
                file_list = [self.imgs_path + 'Others', self.imgs_path + 'Ecklonia', self.imgs_path + 'Padding/Others', self.imgs_path + 'Padding/Ecklonia']
                self.data = []
                for class_path in file_list:
                    class_name = class_path.split("/")[-1]
                    for img_path in glob.glob(class_path + "/*.jpg"):
                        self.data.append([img_path, class_name])
                self.class_map = {"Ecklonia" : 1, "Others": 0}
        else: 
            self.imgs_path = IMG_PATH + str(img_size)+ '_images/'
            file_list = [self.imgs_path + 'Others', self.imgs_path + 'Padding/Others', self.imgs_path + 'Ecklonia', self.imgs_path + 'Padding/Ecklonia']
            self.data = []
            for class_path in file_list:
                class_name = class_path.split("/")[-1]
                for img_path in glob.glob(class_path + "/*.jpg"):
                    self.data.append([img_path, class_name])
            
            # Delete all entries that are used in test_list
            #del_counter = len(self.data)
            #print('Loaded created dataset of {} entries. \nNow deleting {} duplicate entries.'.format(len(self.data), len(test_list[0])))
            if test_list != []: # if test dataset is not extracted out of trainig dataset
                for test_entry in (test_list[0]):
                    if test_entry in self.data:
                        self.data.remove(test_entry)

            #print('Deleted {} duplicate entries'.format(del_counter-len(self.data)))
            self.class_map = {"Ecklonia" : 1, "Others": 0}
        
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        class_id = self.class_map[class_name]
        img = Image.fromarray(img)
        if self.inception:
            train_transforms = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAutocontrast(p=0.5),
            transforms.RandomEqualize(p=0.4),
            transforms.ColorJitter(brightness=0.5, hue=0.2),
            transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            
            ])
        else:
            train_transforms = transforms.Compose([
            transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])
        img_tensor = train_transforms(img)
        #print(type(img_tensor))
        class_id = torch.tensor(class_id)
        return img_tensor, class_id

class KelpClassifier(pl.LightningModule):
    def __init__(self, backbone_name, no_filters, trainer, trial, img_size, batch_size): #dropout, learning_rate, 
        super().__init__()
        # init a pretrained resnet
        self.img_size = img_size
        self.trainer = trainer
        self.pl_module = LightningModule
        self._trial = trial
        self.batch_size = batch_size
        self.monitor = 'f1_score'
        self.hparams.learning_rate = 1e-5
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.backbone_name = backbone_name
        backbone = getattr(models, backbone_name)(weights='DEFAULT')
        #implementing inception_v3

        if self.backbone_name == 'inception_v3': # Initialization for Inception_v3
        #self.model = models.inception_v3(weights='DEFAULT') 
            self.model = backbone
            feature_extract = False
            set_parameter_requires_grad(self.model, feature_extract)
            self.model.aux_logits = False
            self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 2)
            #nn.Linear(10, 2)
            )
        else: # Initialization for all other models
            num_filters = no_filters      
            if num_filters == 0:
                num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
            self.feature_extractor = nn.Sequential(*layers)
            num_target_classes = 2
            self.classifier = nn.Linear(num_filters,  num_target_classes)

        self.training_losses = [[],[],[],[],[],[]]
        self.valid_losses = [[],[],[],[],[],[]]
        self.test_losses = [[],[],[],[],[],[]]
        self.test_eck_p = []
        self.roc_curve = [[],[],[],[]]
        #self.save_hyperparameters(ignore=['trainer','trial'])

    def train_dataloader(self):
        return DataLoader(training_set, batch_size=self.batch_size, num_workers=os.cpu_count(),shuffle=False)
    
    def val_dataloader(self):
        return DataLoader(validation_set, batch_size=self.batch_size, num_workers=os.cpu_count(),shuffle=False)
    
    def on_validation_end(self):
            epoch = self.current_epoch

            current_score = self.trainer.callback_metrics.get(self.monitor)
            
            if current_score is None:
                message = (
                    "The metric '{}' is not in the evaluation logs for pruning. "
                    "Please make sure you set the correct metric name.".format(self.monitor)
                )
                warnings.warn(message)
                return
            
            self._trial.report(current_score, step=epoch)
            if self._trial.should_prune():
                message = "Trial was pruned at epoch {}.".format(epoch)
                # Remove not successful log
                #!
                #shutil.rmtree(self.logger.log_dir, ignore_errors=True)
                raise optuna.TrialPruned(message)

    def forward(self, x):
        x_mlp = x
        if self.backbone_name == 'inception_v3':
            self.model.eval()
            x = self.model(x)
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)
            x = self.classifier(representations)
        
        if MLP_OPT:
            mlp_model = torch.jit.load(MLP_PATH, map_location='cuda')
            mlp_model.eval()
            x_mlp = transforms.CenterCrop(24)(x_mlp)
            x_mlp = mlp_model(x_mlp)
            return x, x_mlp
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        metrics, loss, f1_score, top_eck_p = analyze_pred(self,x, y)
        for i, metric in enumerate(metrics):
            self.training_losses[i].append(metric)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        metrics, loss, f1_score, top_eck_p = analyze_pred(self,x, y)
        for i, metric in enumerate(metrics):
            self.valid_losses[i].append(metric)
        # only log when not sanity checking
        if not(self.trainer.sanity_checking):
            self.log('f1_score', f1_score, on_step=False, on_epoch=True)
        #return f1_score
        image_to_tb(self, batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        metrics, loss, f1_score,top_eck_p = analyze_pred(self,x, y)
        for i, metric in enumerate(metrics):
            self.test_losses[i].append(metric)
        for single_prob in top_eck_p:
            self.test_eck_p.append(single_prob)
        accuracy, precision, recall, f1_score = get_acc_prec_rec_f1(self, metric = self.test_losses)
        #self.log('f1_score', f1_score)
        #return f1_score
    
    def on_train_epoch_end(self):
        #value, var, name
        name = 'train'
        metric = ['loss', 'accuracy', 'precision', 'recall', 'f1_score']
        accuracy, precision, recall, f1_score = get_acc_prec_rec_f1(self, metric = self.training_losses)
        loss = np.mean(self.training_losses[0])

        metric_list = [loss, accuracy,  precision, recall, f1_score]
        for i, var in enumerate(metric):
            log_to_graph(self, metric_list[i], var, name, self.global_step)
        self.training_losses = [[],[],[],[],[],[]]  # reset for next epoch

    def on_validation_epoch_end(self):
        #value, var, name
        name = 'valid'
        metric = ['loss', 'accuracy', 'precision', 'recall', 'f1_score']
        accuracy, precision, recall, f1_score = get_acc_prec_rec_f1(self, metric = self.valid_losses)
        loss = np.mean(self.valid_losses[0])

        metric_list = [loss, accuracy, precision, recall, f1_score]
        if not(self.trainer.sanity_checking):
            for i, var in enumerate(metric):
                log_to_graph(self, metric_list[i], var, name, self.global_step)
        self.valid_losses = [[],[],[],[],[],[]]  # reset for next epoch
    def on_test_epoch_start(self):
        self.roc_curve = [[],[],[],[]]

    def on_test_epoch_end(self):
        #value, var, name
        name = 'test'
        metric = ['loss', 'accuracy', 'precision', 'recall', 'f1_score']
        accuracy, precision, recall, f1_score = get_acc_prec_rec_f1(self, metric = self.test_losses)
        loss = np.mean(self.test_losses[0])
        test_pred_hist = get_pred_histogram(self, self.test_eck_p)
        metric_list = [loss, accuracy, precision, recall, f1_score]
        global path_label
        for i, var in enumerate(metric):
            if not(real_test): #for normal testing
                log_to_graph(self, metric_list[i], var, name, self.global_step)
            else: 
                
                self.logger.experiment.add_scalars('test_'+var, {name: metric_list[i]},path_label)
        
        # Add ROC curve
        #create_roc_curve(self)
        
        # Add probability histogramm to log
        if real_test: prob_name = name+'/'+str(path_label)
        else: prob_name = name
                
        for j, value in enumerate(test_pred_hist):
            step = 5 + j
            log_to_graph(self, value, 'test_probability', prob_name, step)
        self.test = [[],[],[],[],[],[]]  # reset for next epoch
        self.test_eck_p = [] # reset for next epoch
    
    def predict_step(self, batch, batch_idx):
        # This can be used for implementation
        x, y = batch
        y_hat = self(x)
        #loss = F.cross_entropy(y_hat, y)
        prob = F.softmax(y_hat, dim=1)
        top_p, top_class = prob.topk(1, dim = 1)
        
        return batch_idx, int(y[0]), int(top_class.data[0][0]), float(top_p.data[0][0])

    def configure_optimizers(self):
        return getattr(torch.optim, optimizer_name)(self.parameters(), lr=self.hparams.learning_rate)
        #return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

def compare_dir_csv(self):
    my_path = '/pvol/Ecklonia_Database/Original_images/All_Images/'
    # delete every kind of ending that corresponds to jpg in the web address
    self.csv_file_df['file_name'] = self.csv_file_df.point_media_path_best.str.replace(r'(.*)\.(?i)jpg.?', '\\1', regex=True).astype('str')
    # delete everything before the last '/' in the web address
    self.csv_file_df.file_name = self.csv_file_df.file_name.str.replace(r'.*/(.*)', '\\1', regex=True).astype('str')
    # replace all irregular characters with '_' in the web address
    self.csv_file_df.file_name = self.csv_file_df.file_name.str.replace(r"\W", "_", regex=True).astype('str')
    # Add campaign key and .jpg to filename
    self.csv_file_df.file_name = self.csv_file_df.point_media_deployment_campaign_key.str.replace(r"\W", "_", regex=True).astype('str')+'-' + self.csv_file_df.file_name + '.jpg'
    
    # Get all files that are in dir
    onlyfiles = [f for f in listdir(my_path) if isfile(join(my_path, f))]
    dir_list_df = pd.Series(onlyfiles)
    # Check which files are actually in the dir
    self.csv_file_df["exists"] = self.csv_file_df.file_name.isin(dir_list_df)
    print(self.csv_file_df.exists.value_counts())
    # Delete all entries that are not downloaded from CSV file
    self.csv_file_df = self.csv_file_df[self.csv_file_df.exists]



def create_roc_curve(self):
    #TP,TN,FP,FN
    # TP Ecklonia identified Ecklonia
    # TN Others identified Others
    # FP Others identified Ecklonia
    # FN Ecklonia identified Others
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    thresh_roc = [[],[],[],[]]
    tpr =[]
    fpr = []
    for id, threshold in enumerate(range(0,100,5)):
        threshold = float(threshold/100)
        for TP_value in self.roc_curve[0]:
            if TP_value > threshold: thresh_roc[0].append(TP_value)
            else: thresh_roc[3].append(TP_value)
        for FP_value in self.roc_curve[2]:
            if FP_value > threshold: thresh_roc[2].append(FP_value)
            else: thresh_roc[1].append(FP_value)
        for TN_value in self.roc_curve[1]:
            if TN_value > (1-threshold): thresh_roc[1].append(TN_value)
            else: thresh_roc[2].append(TN_value)
        for FN_value in self.roc_curve[3]:
            if FN_value > (1-threshold):thresh_roc[3].append(FN_value)
            else: thresh_roc[0].append(FN_value)
        tpr = len(thresh_roc[0])/(len(thresh_roc[0])+len(thresh_roc[3]))
        fpr = len(thresh_roc[2])/(len(thresh_roc[2])+len(thresh_roc[1]))
        log_to_graph(self, tpr*100, 'test_roc', 'path_label', fpr*100)
        print(tpr*100, fpr*100)
        #self.logger.experiment.add_scalars('test_roc_exp', {'test': fpr*100},'path_label')
        #self.logger.experiment.add_scalars(var, {name: value},fpr*100)
        #self.logger.experiment.add_scalars('test_roc', {name: metric_list[i]},path_label)
    


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_pred_histogram(self, test_eck_p):
    test_pred_hist = [0]*5
    for entry in test_eck_p:
        if (entry.item() >0.5 and entry.item()<=0.6): test_pred_hist[0] += 1
        elif (entry.item() >0.6 and entry.item()<=0.7): test_pred_hist[1] += 1
        elif (entry.item() >0.7 and entry.item()<=0.8): test_pred_hist[2] += 1
        elif (entry.item() >0.8 and entry.item()<=0.9): test_pred_hist[3] += 1
        elif entry.item() >0.9: test_pred_hist[4] += 1
    return test_pred_hist


def get_acc_prec_rec_f1(self, metric):
    TP = np.mean(metric[2]) 
    TN = np.mean(metric[3]) 
    FP = np.mean(metric[4]) 
    FN = np.mean(metric[5])
    
    precision = 0
    recall = 0
    f1_score = 0
    accuracy = 0
    if TP + TN > 0:
        accuracy = (TP + TN)/(TP+FP+TN+FN)
    if TP > 0: 
        precision = TP/(TP+FP)
        recall = TP / (TP + FN)
        f1_score = 2*(precision*recall)/(precision+recall)
    return accuracy, precision, recall, f1_score

def analyze_pred(self,x,y):
        if MLP_OPT:
            y_hat, y_mlp = self(x)
            prob = F.softmax(y_hat, dim=1)
            prob_mlp = F.softmax(y_mlp, dim=1)
            prob = prob*(1-MLP_PERC) + prob_mlp*MLP_PERC
        else:
            y_hat = self(x)
            prob = F.softmax(y_hat, dim=1)
        loss = F.cross_entropy(y_hat, y)
        top_p, top_class = prob.topk(1, dim = 1)
        # Threshold
        global threshold
        top_og = top_p.detach().clone()

        for idx, value in enumerate(top_p):
            if value < threshold: top_p[idx] = 0

        top_eck_p = top_p * top_class
        top_class = torch.reshape(top_class, (-1,))
        accuracy = self.accuracy(top_class, y)
        TP=0
        TN=0
        FP=0
        FN=0
        for i in range(len(top_class)):    
            if top_class[i] == 1:
                if top_class[i] == y[i]: 
                    TP +=1
                    self.roc_curve[0].append(top_og[i].item())
                else: 
                    FP += 1
                    self.roc_curve[2].append(top_og[i].item())
            elif top_class[i] == 0: 
                if top_class[i] == y[i]: 
                    TN += 1
                    self.roc_curve[1].append(top_og[i].item())
                else: 
                    FN += 1
                    self.roc_curve[3].append(top_og[i].item())
            else: print('Error with prediction')
        f1_metric = BinaryF1Score().to('cuda')
        f1_score = f1_metric(top_class, y)
        #prec_metric = BinaryPrecision().to('cuda')
        #prec_score = prec_metric(top_class, y)
        #rec_metric = BinaryRecall().to('cuda')
        #rec_score = rec_metric(top_class, y)
        #metrics=[loss.item(), accuracy.item(), f1_score.item(), prec_score.item(), rec_score.item()]
        metrics=[loss.item(), accuracy.item(), TP, TN, FP, FN]
        return metrics, loss, f1_score, top_eck_p

def get_args():
    parser = argparse.ArgumentParser(description='Enter Parameters to define model training.')

    parser.add_argument('--percent_valid_examples', metavar='pve', type=float, help='The percentage of valid examples taking out of dataset', default=0.1)

    parser.add_argument('--percent_test_examples', metavar='pte', type=float, help='The percentage of test examples taking out of dataset', default=0.1)

    parser.add_argument('--eck_test_perc', metavar='eck_tp', type=float, help='The percentage of ecklonia examples in the test set', default=0.5)

    parser.add_argument('--limit_train_batches', metavar='ltrb', type=float, help='Limits the amount of entries in the trainer for debugging purposes', default=0.1) #!

    parser.add_argument('--limit_val_batches', metavar='lvb', type=float, help='Limits the amount of entries in the trainer for debugging purposes', default=0.1) #!

    parser.add_argument('--limit_test_batches', metavar='lteb', type=float, help='Limits the amount of entries in the trainer for debugging purposes', default=0.1) #!

    parser.add_argument('--epochs', metavar='epochs', type=int, help='The number of epcohs the algorithm trains', default=2)#!

    parser.add_argument('--log_path', metavar='log_path', type=str, help='The path where the logger files are saved', default='/pvol/logs/')

    parser.add_argument('--log_name', metavar='log_name', type=str, help='Name of the experiment.', default='unnamed')

    parser.add_argument('--img_path', metavar='img_path', type=str, help='Path to the database of images', default='/pvol/Ecklonia_Database/')

    parser.add_argument('--n_trials', metavar='n_trials', type=int, help='Number of trials that Optuna runs for', default=None)

    parser.add_argument('--mlp_opt',  help='Defines whether the MLP is activated or not', action='store_true')
    
    parser.add_argument('--mlp_perc', metavar='mlp_perc', type=float, help='Defines the weight that is given to the MLP prediction', default=0.3)

    parser.add_argument('--mlp_path', metavar='mlp_path', type=str, help='Path to the MLP model', default='/home/ubuntu/IMAS/Code/PyTorch/models/81_f1_score.pth')

    parser.add_argument('--backbone', metavar='backbone', type=str, help='Name of the model which should be used for transfer learning', default='inception_v3')

    parser.add_argument('--real_test',  help='If True: a seperate dataset is used, if False dataset is extracted out of training set. ', action='store_false') #!  

    parser.add_argument('--test_img_path', metavar='test_img_path', type=str, help='Path to the database of test images', default='/pvol/Ecklonia_Testbase/NSW_Broughton/')

    parser.add_argument('--img_size', metavar='img_size', type=int, help='Defines the size of the used image.', default=299)

    args =parser.parse_args()
    no_filters = 0
    return args.percent_valid_examples,args.percent_test_examples, args.eck_test_perc,args.limit_train_batches,args.limit_val_batches,args.limit_test_batches,args.epochs,args.log_path,args.log_name,args.img_path, args.n_trials,args.mlp_opt, args.mlp_perc, args.mlp_path,args.backbone, no_filters, args.real_test, args.test_img_path, args.img_size

def log_to_graph(self, value, var, name ,global_step):
    self.logger.experiment.add_scalars(var, {name: value},global_step)

def image_to_tb(self, batch, batch_idx):
    if batch_idx == 0:
        tensorboard = self.logger.experiment
        others_batch_images = []
        ecklonia_batch_images = []
        for index, image in enumerate(batch[0]):
            if batch[1][index] == 0: others_batch_images.append(image) 
            else: ecklonia_batch_images.append(image)
        others_grid = torchvision.utils.make_grid(others_batch_images)
        ecklonia_grid = torchvision.utils.make_grid(ecklonia_batch_images)
        tensorboard.add_image('Ecklonia images', ecklonia_grid, batch_idx)
        tensorboard.add_image('Other images', others_grid, batch_idx)
        tensorboard.close()
        #tensorboard.add_image(batch[0])
    return batch_idx

def get_test_dataset(img_size, PERCENT_TEST_EXAMPLES, ECK_TEST_PERC):
    unpad_path = IMG_PATH + str(img_size)+ '_images/'
    pads_path = IMG_PATH + str(img_size)+ '_images/Padding/'
    both_path = [unpad_path, pads_path]
    unpad_file_list = [unpad_path + 'Others', unpad_path + 'Ecklonia']
    pad_file_list =  [pads_path + 'Others', pads_path + 'Ecklonia']
    both_file_list = [unpad_file_list, pad_file_list]
    perc = [PERCENT_TEST_EXAMPLES * (1-ECK_TEST_PERC)*100, 1]
    unpad_data = []
    pad_data = []
    both_data = [unpad_data, pad_data]
    perc_id = 0
    counter = [[0,0],[0,0]]
    if int(PERCENT_TEST_EXAMPLES * ECK_TEST_PERC * 100) ==0: perc[1] = 1
    else: perc[1] = int(PERCENT_TEST_EXAMPLES * ECK_TEST_PERC * 100)
    for idx in range(2):
        # First loop iterates over unpad files, second over padded files
        for class_path in both_file_list[idx]:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                #only add path to data if PERCENT_TEST_EXAMPLES allows
                if class_name == 'Others': 
                    perc_id = 0
                    class_id = 0
                else: 
                    perc_id = 1
                    class_id = 1
                if random.randint(0,99) < perc[perc_id]:
                    both_data[idx].append([img_path, class_name])
                    counter[idx][class_id] += 1
    return both_data

def display_testset(test_loader):
    counter = [[0,0],[0,0]] # First [] is Others, then sub [] is unpadded, padded
    pad_idx = 0
    for i in range(len(test_loader.dataset.data)):
        if 'Padding' in test_loader.dataset.data[i][0]: 
            pad_idx = 1
        else: pad_idx = 0
        if test_loader.dataset.data[i][1] == 'Others':
            class_id = 0
        else: class_id = 1
        counter[class_id][pad_idx] += 1
    print('The Test-Dataset comprises of: \nUniform Ecklonia {}\nUniform Others {} \nPadded Ecklonia {}\nPadded Others {}'.format(counter[1][0], counter[0][0], counter[1][1], counter[0][1]))
    #print(test_loader.dataset.data[1][1])

def objective(trial: optuna.trial.Trial) -> float:

    ###############
    # Optuna Params
    ###############
    #dropout = trial.suggest_float("dropout", 0.2, 0.5)
    BATCHSIZE = 64#trial.suggest_int("batchsize", 8, 128)
    '''
    LEARNING_RATE = trial.suggest_float(
        "learning_rate_init", 1e-8, 1e-1, log=True
    ) #min needs to be 1e-6
    ''' 
    #LEARNING_RATE = 0.0000050000
    global optimizer_name
    #optimizer_name = trial.suggest_categorical("optimizer", ['Adam', 'Adagrad', 'Adadelta', 'Adamax', 'AdamW', 'ASGD', 'NAdam', 'RAdam', 'RMSprop', 'Rprop', 'SGD'])
    optimizer_name = 'SGD'
    global MLP_PERC, rvf_perc, rhf_perc, rauto_perc, requa_perc, rbright_perc, rhue_perc
    MLP_PERC = trial.suggest_float("mlp_perc", 0, 1) 
    global threshold
    threshold = trial.suggest_float("threshold", 0.5, 1)
    #rvf_perc = trial.suggest_float("rvf_perc", 0, 1)
    #rhf_perc = trial.suggest_float("rhf_perc", 0, 1)
    #rauto_perc = trial.suggest_float("rauto_perc", 0, 1)
    #requa_perc = trial.suggest_float("requa_perc", 0, 1)
    #rbright_perc = trial.suggest_float("rbright_perc", 0, 1)
    #rhue_perc = trial.suggest_float("rhue_perc", 0, 0.5)

    #img_size = trial.suggest_categorical("img_size", [  
                    #"256", 
                    #"288",
                    #'299',
                    #'304', 
                    #"320", 
                    #'400', 
                    #'448',
                    #'480', 
                    #'512', 
                    #'544',
                    #'576',
                    #'608' 
    #                ])
    ##############
    # Data Loading
    ##############
    inception = False
    if backbone_name == 'inception_v3': 
        inception = True
        TEST_BATCHSIZE = BATCHSIZE
    else: TEST_BATCHSIZE = 1 # Batchsize scaled down to one because of diff image sizes 
    if real_test:
        test_list = []
    else: 
        test_list = get_test_dataset(img_size, PERCENT_TEST_EXAMPLES, ECK_TEST_PERC)
    
    train_val_set = CSV_Dataset(img_size, test_list, test = False, inception = inception, test_img_path = test_img_path)
    
    global training_set, validation_set
    training_set, validation_set = torch.utils.data.random_split(train_val_set,[0.90, 0.10], generator=torch.Generator().manual_seed(123))

    # Create data loaders for our datasets; shuffle for training and for validation
    #train_loader = torch.utils.data.DataLoader(training_set, BATCHSIZE, shuffle=True, num_workers=os.cpu_count())
    
    #val_loader = torch.utils.data.DataLoader(validation_set, BATCHSIZE, shuffle=False, num_workers=os.cpu_count())


    acc_val = 'cpu'
    if torch.cuda.is_available(): acc_val = 'gpu'
    trainer = pl.Trainer(
        logger=True,
        default_root_dir=LOGGER_PATH+LOG_NAME+'/'+str(img_size)+'/',
        enable_checkpointing=True,
        max_epochs=EPOCHS,
        accelerator=acc_val,
        callbacks=[EarlyStopping(monitor="f1_score", mode="max")],
        limit_train_batches=LIMIT_TRAIN_BATCHES,
        limit_val_batches=LIMIT_VAL_BATCHES,
        limit_test_batches=LIMIT_TEST_BATCHES,
        precision=16,
        log_every_n_steps=50,
        #auto_scale_batch_size="power",
        #auto_lr_find=True,
    )
    
    model = KelpClassifier(backbone_name, 
                            no_filters, 
                            #LEARNING_RATE, 
                            batch_size = 128, 
                            trial = trial, 
                            #monitor ='f1_score', 
                            trainer= trainer,  
                            #pl_module = LightningModule, 
                            img_size=img_size)
    #dropout=dropout,
    #trainer.tune(model, train_loader,val_loader )
    #trainer.fit(model, train_loader,val_loader )
    trainer.tune(model)
    trainer.fit(model)
    

    ##########
    # Logger #
    ##########
    hyperparameters = dict(
            optimizer_name = optimizer_name,
            learning_rate=model.hparams.learning_rate, #LEARNING_RATE, 
            image_size = img_size, 
            backbone = backbone_name,
            #batchsize = model.batch_size,
            #rvf_perc = rvf_perc, 
            #rhf_perc = rhf_perc, 
            #rauto_perc = rauto_perc, 
            #requa_perc=requa_perc, 
            #rbright_perc=rbright_perc, 
            #lrhue_perc=rhue_perc, 
            threshold = threshold,
            f1_score = trainer.callback_metrics["f1_score"].item())

    if MLP_OPT:
        hyperparameters['mlp_percentage']=MLP_PERC 
    if not(real_test):
        hyperparameters['test_perc'] = PERCENT_TEST_EXAMPLES 
        hyperparameters['eck_test_perc'] = ECK_TEST_PERC


    trainer.logger.log_hyperparams(hyperparameters)

    
    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    # value copied here so that it is not influenced by different testing options
    f1_score_end = trainer.callback_metrics["f1_score"].item()

    if real_test:
        path_list = [
            '/pvol/Ecklonia_Testbase/WA/', 
            '/pvol/Ecklonia_Testbase/NSW_Broughton/', 
            '/pvol/Ecklonia_Testbase/VIC_Prom/',
            '/pvol/Ecklonia_Testbase/VIC_Discoverybay/', 
            '/pvol/Ecklonia_Testbase/TAS_Lanterns/']
        for idx ,path in enumerate(path_list):
            global path_label 
            #path_label = re.sub(".*/(.*)/", "\\1", path)
            path_label = idx
            test_set = GeneralDataset(img_size, test_list, test = True, inception=inception, test_img_path = path)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=TEST_BATCHSIZE, shuffle=False, num_workers=os.cpu_count())
            display_testset(test_loader)
            trainer.test(ckpt_path='best', dataloaders=test_loader)
    else: 
        test_set = GeneralDataset(img_size, test_list, test = True, inception=inception,test_img_path = test_img_path)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=TEST_BATCHSIZE, shuffle=False, num_workers=os.cpu_count())
        display_testset(test_loader)
        trainer.test(ckpt_path='best', dataloaders=test_loader)
    
    print('Number of samples overall: {}'.format(len(train_val_set) + len(test_set)))
    
    return f1_score_end

if __name__ == '__main__':
    PERCENT_VALID_EXAMPLES, PERCENT_TEST_EXAMPLES, ECK_TEST_PERC, LIMIT_TRAIN_BATCHES, LIMIT_VAL_BATCHES, LIMIT_TEST_BATCHES, EPOCHS, LOGGER_PATH, LOG_NAME, IMG_PATH, N_TRIALS, MLP_OPT, MLP_PERC, MLP_PATH, backbone_name, no_filters, real_test, test_img_path, img_size = get_args()
    
    test_log_count = 0 # Needed to display all five datasets

    if MLP_OPT: print('MLP is activated')
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_TRIALS, timeout=None)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    #cli_main()