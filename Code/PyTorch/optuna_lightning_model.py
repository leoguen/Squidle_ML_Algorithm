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
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

writer = SummaryWriter()

class CSV_Dataset(Dataset):
    def __init__(self, img_size, test_list, test, inception, csv_data_path):
        self.test_indicator = test
        self.csv_data_path = csv_data_path
        self.inception = inception
        self.csv_file_df = pd.read_csv(csv_data_path)
        self.class_map = {one_word_label : 1, "Others": 0}
        compare_dir_csv(self, csv_data_path)
        # Add unpadded and padded entries to data

    def __len__(self):
        #print(self.csv_file_df.shape[0])
        return self.csv_file_df.shape[0]    
    
    def __getitem__(self, idx):
        img_path = str(re.sub(r'(.*)/.*', '\\1', self.csv_data_path)) + '/All_Images/' +self.csv_file_df.file_name.iloc[idx]
        class_id = torch.tensor(0)
        if label_name == 'Ecklonia radiata':
            if self.csv_file_df.label_name.iloc[idx] == label_name: 
                class_id = torch.tensor(1)
        else: #If translated labels are used
            if self.csv_file_df.label_translated_name.iloc[idx] == label_name: 
                class_id = torch.tensor(1) 
        img = cv2.imread(img_path)
        #class_id = self.class_map[class_name]
        img = Image.fromarray(img)
        x = self.csv_file_df.point_x.iloc[idx] # At this point only float
        y = self.csv_file_df.point_y.iloc[idx] # At this point only float
        #print(img.size)
        x_img, y_img = img.size
        loc_img_size = img_size
        if crop_perc != 0:
            loc_img_size = int((x_img + y_img)/2 * crop_perc)
        x = x_img*x #Center position
        y = y_img*y #Center position
        
        # get crop coordinates
        x0, x1, y0, y1 = get_crop_points(self, x, y, img, loc_img_size)
        cropped_img = img.crop((x0, y0, x1, y1))
        x_perc = (x-x0)/(x1-x0)
        y_perc = (y-y0)/(y1-y0)
        x_crop_size, y_crop_size = cropped_img.size
        x_crop =  x_perc
        y_crop =  y_perc
        if self.inception:
            if self.test_indicator: 
                train_transforms = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ])
            else:
                train_transforms = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomAutocontrast(p=0.5),
                #transforms.RandomEqualize(p=0.4),
                #transforms.ColorJitter(brightness=0.5, hue=0.2),
                transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])
        else:
            train_transforms = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])
        img_tensor = train_transforms(cropped_img)

        return img_tensor, class_id , x_crop, y_crop

class GeneralDataset(Dataset):
    def __init__(self, img_size, test_list, test, inception, test_img_path):
        self.inception = inception
        self.test_indicator = test
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
            if self.test_indicator: 
                train_transforms = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ])
            else:
                train_transforms = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAutocontrast(p=0.5),
                #transforms.RandomEqualize(p=0.4),
                transforms.ColorJitter(brightness=0.5, hue=0.2),
                transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
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
        self.csv_test_results = [[],[],[],[],[]]
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
        train_dataloader = DataLoader(training_set, batch_size=self.batch_size, num_workers=60)#os.cpu_count(),shuffle=False)
        #display_dataloader(train_dataloader, 'train_dataloader')
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(validation_set, batch_size=self.batch_size, num_workers=60)#os.cpu_count(),shuffle=False)
        #display_dataloader(val_dataloader, 'val_dataloader')
        return val_dataloader
        #return DataLoader(validation_set, batch_size=self.batch_size, num_workers=os.cpu_count(),shuffle=False)
    
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
            #if self._trial.should_prune():
                #message = "Trial was pruned at epoch {}.".format(epoch)
                # Remove not successful log
                #!
                #shutil.rmtree(self.logger.log_dir, ignore_errors=True)
                #raise optuna.TrialPruned(message) #!

    def forward(self, x, x_crop, y_crop):
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
            #! add crop here
            mlp_model = torch.jit.load(MLP_PATH, map_location='cuda')
            mlp_model.eval()
            x_mlp_24 = torch.empty((len(x_mlp),3,24,24), dtype=torch.float32).cuda()
            for idx in range(len(x_mlp)):
                x_mlp_24[idx] = transforms.functional.crop(x_mlp[idx], int(y_crop[idx].item()*299) + 12, int(x_crop[idx].item()*299)-12, 24, 24)
            x_mlp = mlp_model(x_mlp_24)
            return x, x_mlp
        return x

    def training_step(self, batch, batch_idx):
        try:
            x, y, x_crop, y_crop = batch
        except: 
            x, y = batch
            x_crop = 0.5 #! Added for GeneralDataset 
            y_crop = 0.5 #!
        #x, y = batch
        metrics, loss, f1_score, top_eck_p,top_class, res_y, prob = analyze_pred(self,x, y, x_crop, y_crop)
        for i, metric in enumerate(metrics):
            self.training_losses[i].append(metric)
        return loss
    
    def validation_step(self, batch, batch_idx):
        try:
            x, y, x_crop, y_crop = batch
        except: 
            x, y = batch
            x_crop = 0.5 #! Added for GeneralDataset 
            y_crop = 0.5 #!
        #x, y = batch
        metrics, loss, f1_score, top_eck_p, top_class, res_y, prob = analyze_pred(self,x, y, x_crop, y_crop)
        for i, metric in enumerate(metrics):
            self.valid_losses[i].append(metric)
        # only log when not sanity checking
        if not(self.trainer.sanity_checking):
            self.log('f1_score', f1_score, on_step=False, on_epoch=True)
        #return f1_score
        image_to_tb(self, batch, batch_idx, 'train')
    
    def test_step(self, batch, batch_idx):
        try:
            x, y, x_crop, y_crop = batch
        except: 
            x, y = batch
            x_crop = 0.5 #! Added for GeneralDataset 
            y_crop = 0.5 #!
        metrics, loss, f1_score,top_eck_p, top_class, res_y, prob = analyze_pred(self,x, y, x_crop, y_crop)
        #metrics=[loss.item(), accuracy.item(), TP, TN, FP, FN]
        for i, metric in enumerate(metrics):
            self.test_losses[i].append(metric)
        for single_prob in top_eck_p:
            self.test_eck_p.append(single_prob)
        accuracy, precision, recall, f1_score = get_acc_prec_rec_f1(self, metric = self.test_losses)
        image_to_tb(self, batch, batch_idx, 'test')
        
        global path_label
        test_results = torch.cat((prob, y.unsqueeze(dim=1)), dim=1)#.cpu().numpy()
        test_results = torch.cat((test_results, top_class.unsqueeze(dim=1)), dim=1).cpu().numpy()
        #T = torch.cat((prob,y), -1)
        self.csv_test_results[path_label].extend(test_results)

        #self.csv_test_results[path_label] = torch.cat((self.csv_test_results[path_label], test_results.unsqueeze(dim=1)), dim=1)
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
        
        plot_confusion_matrix(self)
        # Add ROC curve
        #create_roc_curve(self)
        
        # Add probability histogramm to log
        if real_test: prob_name = name+'/'+str(path_label)
        else: prob_name = name
                
        for j, value in enumerate(test_pred_hist):
            step = 5 + j
            log_to_graph(self, value, 'test_probability', prob_name, step)
        self.test_losses = [[],[],[],[],[],[]]  # reset for next epoch
        self.test_eck_p = [] # reset for next epoch
        
        save_test_csv(self)
    
    def predict_step(self, batch, batch_idx):
        # This can be used for implementation
        x, y, x_crop, y_crop = batch
        #x, y = batch
        y_hat = self(x)
        #loss = F.cross_entropy(y_hat, y)
        prob = F.softmax(y_hat, dim=1)
        top_p, top_class = prob.topk(1, dim = 1)
        
        return batch_idx, int(y[0]), int(top_class.data[0][0]), float(top_p.data[0][0])

    def configure_optimizers(self):
        return getattr(torch.optim, optimizer_name)(self.parameters(), lr=self.hparams.learning_rate)
        #return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

def save_test_csv(self):
    global path_label
    
    path_list = [
    'WA', 
    'NSW_Broughton', 
    'VIC_Prom',
    'VIC_Discoverybay', 
    'TAS_Lanterns']
    
    if crop_perc != 0:
        dir = LOGGER_PATH+LOG_NAME+'/perc_'+str(int(crop_perc*100))+'/' 
    else: 
        dir = LOGGER_PATH+LOG_NAME+'/'+str(img_size)+'/'

    test_csv_path = dir + 'lightning_logs/version_{}/'.format(self.trainer.logger.version) + 'test_results_'+ path_list[path_label]+'.csv'

    df = pd.DataFrame(self.csv_test_results[path_label], columns=('Others', 'Ecklonia', 'Truth', 'Pred')) 
    
    # saving the dataframe 
    #df.to_csv(LOGGER_PATH + LOG_NAME +'/'+ str(img_size)+'/lightnings_logs/test_results_'+ path_list[path_label]+'.csv')
    df.to_csv(test_csv_path)
    

def plot_confusion_matrix(self):
    #! self.test has TP on [2], TN on [3], FP on [4], FN on [5]
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    np_values = np.array([['Ecklonia', np.sum(self.test_losses[2]), np.sum(self.test_losses[4])], ['Others', np.sum(self.test_losses[5]), np.sum(self.test_losses[3])]])
    
    df = pd.DataFrame(np_values, columns=['Pred/True', 'Ecklonia', 'Others'])

    the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    the_table[(1, 1)].set_facecolor("#B0FAA4")
    the_table[(2, 2)].set_facecolor("#B0FAA4")
    the_table[(1, 2)].set_facecolor("#F06969")
    the_table[(2, 1)].set_facecolor("#F06969")
    the_table.set_fontsize(30)
    the_table.auto_set_column_width(col=list(range(3)))
    the_table.scale(4,4)
    #table[(1, 0)].set_facecolor("B0FAA4")
    #F06969

    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #plt.show()
    tensorboard = self.logger.experiment
    tensorboard.add_image('Confusion Matrix Test' , data, path_label, dataformats='HWC')


def get_crop_points(self, x, y, original_image, img_size):
    x_img, y_img = original_image.size
    crop_dist = img_size/2 
    if x - crop_dist < 0: x0 = 0
    else: x0 = x - crop_dist

    if y - crop_dist < 0: y0 = 0
    else: y0 = y - crop_dist

    if x + crop_dist > x_img: x1 = x_img
    else: x1 = x + crop_dist

    if y + crop_dist > y_img: y1 = y_img
    else: y1 = y + crop_dist

    return  int(x0), int(x1), int(y0), int(y1)

def compare_dir_csv(self, csv_path):
    #csv_path = csv_path.replace(r'(.*)/.*', '\\1', regex=True).astype('str')
    img_path = str(re.sub(r'(.*)/.*', '\\1', csv_path)) + '/All_Images/' 
    # delete every kind of ending that corresponds to jpg in the web address
    self.csv_file_df.columns = self.csv_file_df.columns.str.replace('[.]', '_')
    self.csv_file_df['file_name'] = self.csv_file_df.point_media_path_best.str.replace(r'(.*)\.(?i)jpg.?', '\\1', regex=True).astype('str')
    # delete everything before the last '/' in the web address
    self.csv_file_df.file_name = self.csv_file_df.file_name.str.replace(r'.*/(.*)', '\\1', regex=True).astype('str')
    # replace all irregular characters with '_' in the web address
    self.csv_file_df.file_name = self.csv_file_df.file_name.str.replace(r"\W", "_", regex=True).astype('str')
    # Add campaign key and .jpg to filename
    self.csv_file_df.file_name = self.csv_file_df.point_media_deployment_campaign_key.str.replace(r"\W", "_", regex=True).astype('str')+'-' + self.csv_file_df.file_name + '.jpg'
    
    # Get all files that are in dir
    onlyfiles = [f for f in listdir(img_path) if isfile(join(img_path, f))]
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
    TP = np.sum(metric[2]) 
    TN = np.sum(metric[3]) 
    FP = np.sum(metric[4]) 
    FN = np.sum(metric[5])
    
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

def analyze_pred(self,x,y, x_crop, y_crop):
        if MLP_OPT:
            y_hat, y_mlp = self(x, x_crop, y_crop)
            prob = F.softmax(y_hat, dim=1)
            prob_mlp = F.softmax(y_mlp, dim=1)
            prob = prob*(1-MLP_PERC) + prob_mlp*MLP_PERC
        else:
            y_hat = self(x, x_crop, y_crop)
            prob = F.softmax(y_hat, dim=1)
        loss = F.cross_entropy(y_hat, y)
        top_p, top_class = prob.topk(1, dim = 1)
        # Threshold
        global threshold
        top_og = top_p.detach().clone()

        #for idx, value in enumerate(top_p):
        #    if value < threshold: top_p[idx] = 0

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
        return metrics, loss, f1_score, top_eck_p, top_class, y, prob

def get_args():
    parser = argparse.ArgumentParser(description='Enter Parameters to define model training.')

    parser.add_argument('--percent_valid_examples', metavar='pve', type=float, help='The percentage of valid examples taking out of dataset', default=0.1)

    parser.add_argument('--percent_test_examples', metavar='pte', type=float, help='The percentage of test examples taking out of dataset', default=0.1)

    parser.add_argument('--eck_test_perc', metavar='eck_tp', type=float, help='The percentage of ecklonia examples in the test set', default=1.0)

    parser.add_argument('--limit_train_batches', metavar='ltrb', type=float, help='Limits the amount of entries in the trainer for debugging purposes', default=0.1) #!

    parser.add_argument('--limit_val_batches', metavar='lvb', type=float, help='Limits the amount of entries in the trainer for debugging purposes', default=0.1) #!

    parser.add_argument('--limit_test_batches', metavar='lteb', type=float, help='Limits the amount of entries in the trainer for debugging purposes', default=0.5) #!

    parser.add_argument('--epochs', metavar='epochs', type=int, help='The number of epcohs the algorithm trains', default=3) #!

    parser.add_argument('--log_path', metavar='log_path', type=str, help='The path where the logger files are saved', default='/pvol/logs/')

    parser.add_argument('--log_name', metavar='log_name', type=str, help='Name of the experiment.', default='unnamed')

    parser.add_argument('--img_path', metavar='img_path', type=str, help='Path to the database of images', default='/pvol/Ecklonia_1_to_10_Database/') #/pvol/Seagrass_Database/

    parser.add_argument('--csv_path', metavar='csv_path', type=str, help='Path to the csv file describing the images', default='/pvol/Ecklonia_1_to_10_Database/Original_images/588335_1_to_10_Ecklonia_radiata.csv')
    #/pvol/Ecklonia_Database/Original_images/106704_normalized_deployment_key_list.csv
    #/pvol/Seagrass_Database/Original_images/14961_Seagrass_cover_NMSC_list.csv
    #/pvol/Ecklonia_1_to_10_Database/Original_images/588335_1_to_10_Ecklonia_radiata_NMSC_list.csv

    parser.add_argument('--n_trials', metavar='n_trials', type=int, help='Number of trials that Optuna runs for', default=1) #!

    parser.add_argument('--mlp_opt',  help='Defines whether the MLP is activated or not', action='store_true')
    
    parser.add_argument('--mlp_perc', metavar='mlp_perc', type=float, help='Defines the weight that is given to the MLP prediction', default=0.3)

    parser.add_argument('--mlp_path', metavar='mlp_path', type=str, help='Path to the MLP model', default='/home/ubuntu/IMAS/Code/PyTorch/models/81_f1_score.pth')

    parser.add_argument('--backbone', metavar='backbone', type=str, help='Name of the model which should be used for transfer learning', default='inception_v3')

    parser.add_argument('--real_test',  help='If True: a seperate dataset is used, if False dataset is extracted out of training set. ', action='store_false') #!

    parser.add_argument('--test_img_path', metavar='test_img_path', type=str, help='Path to the database of test images', default='/pvol/Ecklonia_Testbase/NSW_Broughton/')

    parser.add_argument('--img_size', metavar='img_size', type=int, help=
    'Defines the size of the used image.', default=299)
    
    parser.add_argument('--crop_perc', metavar='crop_perc', type=int, help= 'Defines percentage that is used to crop the image.', default=0.0)

    parser.add_argument('--batch_size', metavar='batch_size', type=int, help= 'Defines batch_size that is used to train algorithm.', default=32) #!

    parser.add_argument('--label_name', metavar='label_name', type=str, help='Name of the label used in the csv file', default='Ecklonia radiata') #Seagrass cover
    
    args =parser.parse_args()
    no_filters = 0
    return args.percent_valid_examples,args.percent_test_examples, args.eck_test_perc,args.limit_train_batches,args.limit_val_batches,args.limit_test_batches,args.epochs,args.log_path,args.log_name,args.img_path, args.n_trials,args.mlp_opt, args.mlp_perc, args.mlp_path,args.backbone, no_filters, args.real_test, args.test_img_path, args.img_size, args.csv_path, args.crop_perc, args.batch_size, args.label_name

def log_to_graph(self, value, var, name ,global_step):
    self.logger.experiment.add_scalars(var, {name: value},global_step)

def image_to_tb(self, batch, batch_idx, step_name):
    global path_label
    if batch_idx == 0:
        tensorboard = self.logger.experiment
        others_batch_images = []
        ecklonia_batch_images = []
        for index, image in enumerate(batch[0]):
            image = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])(image)
            #unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
            if batch[1][index] == 0: others_batch_images.append(image) 
            else: ecklonia_batch_images.append(image)
        
        if step_name == 'test':
            others_grid = torchvision.utils.make_grid(others_batch_images)
            tensorboard.add_image('Other images ' +step_name, others_grid, path_label)
            try: 
                ecklonia_grid = torchvision.utils.make_grid(ecklonia_batch_images)
                tensorboard.add_image(one_word_label+' images '+step_name, ecklonia_grid, path_label)
            except:
                print('No '+one_word_label+' entries to post to Tensorboard.')
            tensorboard.close()
        else: 
            others_grid = torchvision.utils.make_grid(others_batch_images)
            tensorboard.add_image('Other images ' +step_name, others_grid, batch_idx)
            try: 
                ecklonia_grid = torchvision.utils.make_grid(ecklonia_batch_images)
                tensorboard.add_image(one_word_label + ' images '+step_name, ecklonia_grid, batch_idx)
            except:
                print('No '+ one_word_label+' entries to post to Tensorboard.')
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

def display_dataloader(data_loader, name):
    counter = [0,0]
    
    #Used for General Dataset 
    '''
    #print(data_loader.dataset.data[:][1])
    for entry in data_loader.dataset.data:
        label = entry[1]
        if label == one_word_label: 
            counter[1] +=1
        else:
            counter[0] +=1
    print('The {} comprises of: {} {} & Others {} \n'.format(name, label_name, counter[1], counter[0]))
    '''
    #Used for CSV Dataset
    
    for idx in data_loader.dataset.indices:
        if label_name == 'Ecklonia radiata':
            if data_loader.dataset.dataset.csv_file_df.label_name.iloc[idx] == label_name:
                #print('Eck Label ' + data_loader.dataset.dataset.csv_file_df.label_name[idx])
                counter[1] += 1 
            else: 
                #print('Others label '+ (data_loader.dataset.dataset.csv_file_df.label_name[idx]))
                counter[0] += 1
        else: #Used for translated label name
            if data_loader.dataset.dataset.csv_file_df.label_translated_name.iloc[idx] == label_name:
                #print('Eck Label ' + data_loader.dataset.dataset.csv_file_df.label_name[idx])
                counter[1] += 1 
            else: 
                #print('Others label '+ (data_loader.dataset.dataset.csv_file_df.label_name[idx]))
                counter[0] += 1
    print('The {} comprises of: {} {} & Others {} \n'.format(name, label_name, counter[1], counter[0]))
    
    
    '''
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
'''

def objective(trial: optuna.trial.Trial) -> float:

    ###############
    # Optuna Params
    ###############
    #dropout = trial.suggest_float("dropout", 0.2, 0.5)
    #trial.suggest_int("batchsize", 8, 128)
    '''
    LEARNING_RATE = trial.suggest_float(
        "learning_rate_init", 1e-8, 1e-1, log=True
    ) #min needs to be 1e-6
    ''' 
    #LEARNING_RATE = 0.0000050000
    global optimizer_name
    #optimizer_name = trial.suggest_categorical("optimizer", ['Adam', 'Adagrad', 'Adadelta', 'Adamax', 'AdamW', 'ASGD', 'NAdam', 'RAdam', 'RMSprop', 'Rprop', 'SGD'])
    optimizer_name = 'Adam'
    global rvf_perc, rhf_perc, rauto_perc, requa_perc, rbright_perc, rhue_perc
    if MLP_OPT:
        global MLP_PERC 
        MLP_PERC = trial.suggest_float("mlp_perc", 0, 1) 
    
    global threshold
    #threshold = trial.suggest_float("threshold", 0.5, 1)
    threshold = 0.5
    
    #global img_size
    #img_size = trial.suggest_int("img_size", 16, 2048, step=16)

    ##############
    # Data Loading
    ##############
    inception = False
    if backbone_name == 'inception_v3': 
        inception = True

    if real_test:
        test_list = []
    else: 
        test_list = get_test_dataset(img_size, PERCENT_TEST_EXAMPLES, ECK_TEST_PERC)

    #train_val_set = GeneralDataset(img_size, test_list, test = False, inception = inception, test_img_path = csv_path)
    train_val_set = CSV_Dataset(img_size, test_list, test = False, inception = inception, csv_data_path = csv_path)
    
    global training_set, validation_set
    training_set, validation_set = torch.utils.data.random_split(train_val_set,[0.90, 0.10], generator=torch.Generator().manual_seed(423))

    # Create data loaders for our datasets; shuffle for training and for validation
    #train_loader = torch.utils.data.DataLoader(training_set, BATCHSIZE, shuffle=True, num_workers=os.cpu_count())
    
    #val_loader = torch.utils.data.DataLoader(validation_set, BATCHSIZE, shuffle=False, num_workers=os.cpu_count())

    acc_val = 'cpu'
    if torch.cuda.is_available(): acc_val = 'gpu'
    
    if crop_perc != 0:
        dir = LOGGER_PATH+LOG_NAME+'/perc_'+str(int(crop_perc*100))+'/' 
    else: 
        dir = LOGGER_PATH+LOG_NAME+'/'+str(img_size)+'/'

    trainer = pl.Trainer(
        logger=True,
        default_root_dir=dir,
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
        auto_lr_find=False, #!
    )
    
    model = KelpClassifier(backbone_name, 
                            no_filters, 
                            #LEARNING_RATE, 
                            batch_size = batch_size, 
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
    if crop_perc != 0:
        hyperparameters['crop_perc']=crop_perc
    else:
        hyperparameters['image_size']= img_size
    if MLP_OPT:
        hyperparameters['mlp_percentage']=MLP_PERC 
    if not(real_test):
        hyperparameters['test_perc'] = PERCENT_TEST_EXAMPLES 
        hyperparameters['eck_test_perc'] = ECK_TEST_PERC


    trainer.logger.log_hyperparams(hyperparameters)

    
    # Handle pruning based on the intermediate value.
    #if trial.should_prune(): #!
        #raise optuna.TrialPruned()
    
    # value copied here so that it is not influenced by different testing options
    f1_score_end = trainer.callback_metrics["f1_score"].item()

    if real_test:
        #Used for CSV Dataset
        
        path_list = [
            '/pvol/Ecklonia_Testbase/WA/Original_images/annotations-u45-leo_kelp_SWC_WA_AI_test-leo_kelp_AI_SWC_WA_test_25pts-8148-7652a9b48f0e3186fe5d-dataframe.csv', 
            '/pvol/Ecklonia_Testbase/NSW_Broughton/Original_images/annotations-u45-leo_kelp_AI_test_broughton_is_NSW-leo_kelp_AI_test_broughton_is_25pts-8152-7652a9b48f0e3186fe5d-dataframe.csv', 
            '/pvol/Ecklonia_Testbase/VIC_Prom/Original_images/annotations-u45-leo_kelp_AI_test_prom_VIC-leo_kelp_AI_test_prom_25pts-8150-7652a9b48f0e3186fe5d-dataframe.csv',
            '/pvol/Ecklonia_Testbase/VIC_Discoverybay/Original_images/annotations-u45-leo_kelp_AI_test_discoverybay_VIC_phylospora-leo_kelp_AI_test_db_phylospora_25pts-8149-7652a9b48f0e3186fe5d-dataframe.csv', 
            '/pvol/Ecklonia_Testbase/TAS_Lanterns/Original_images/annotations-u45-leo_kelp_AI_test_lanterns_TAS-leo_kelp_AI_test_lanterns_25pts-8151-7652a9b48f0e3186fe5d-dataframe.csv']
        
        '''
        #Used for Generaldataset
        path_list = [
            '/pvol/Ecklonia_Testbase/WA/', 
            '/pvol/Ecklonia_Testbase/NSW_Broughton/', 
            '/pvol/Ecklonia_Testbase/VIC_Prom/',
            '/pvol/Ecklonia_Testbase/VIC_Discoverybay/', 
            '/pvol/Ecklonia_Testbase/TAS_Lanterns/']
        '''
        for idx ,path in enumerate(path_list):
            global path_label 
            #path_label = re.sub(".*/(.*)/", "\\1", path)
            path_label = idx
            test_set = CSV_Dataset(img_size, test_list, test = True, inception=inception, csv_data_path= path)
            #test_set = GeneralDataset(img_size, test_list, test=True, inception=inception, test_img_path=path)
            # Just looks at one dataset
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=60)#os.cpu_count())
            #display_dataloader(test_loader, 'Test Loader'+str(path_label))
            trainer.test(ckpt_path='best', dataloaders=test_loader)
    else: 
        test_set = GeneralDataset(img_size, test_list, test = True, inception=inception,test_img_path = test_img_path)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=60)#os.cpu_count())
        #display_dataloader(test_loader, 'Test Loader Generaldataset')
        trainer.test(ckpt_path='best', dataloaders=test_loader)
    
    print('Number of samples overall: {}'.format(len(train_val_set) + len(test_set)))
    
    return f1_score_end

if __name__ == '__main__':
    PERCENT_VALID_EXAMPLES, PERCENT_TEST_EXAMPLES, ECK_TEST_PERC, LIMIT_TRAIN_BATCHES, LIMIT_VAL_BATCHES, LIMIT_TEST_BATCHES, EPOCHS, LOGGER_PATH, LOG_NAME, IMG_PATH, N_TRIALS, MLP_OPT, MLP_PERC, MLP_PATH, backbone_name, no_filters, real_test, test_img_path, img_size, csv_path, crop_perc, batch_size, label_name = get_args()

    one_word_label = 'One_Word_Label_Not_Defined'
    if label_name == 'Ecklonia radiata':
        one_word_label = 'Ecklonia'
    elif label_name == 'Seagrass cover':
        one_word_label = 'Seagrass'
    elif label_name == 'Hard coral cover':
        one_word_label = 'Hardcoral'
    elif label_name == 'Macroalgal canopy cover':
        one_word_label = 'Macroalgal'
    
    #model_specs = [
        #['resnet50', 0],
        #['googlenet', 0], 
        #['convnext_large', 1536], 
        #['convnext_small', 768], 
        #['resnext101_64x4d', 0], 
        #['efficientnet_v2_l', 1280], 
        #['vit_h_14', 1280], #does not work  
        #['regnet_x_32gf', 0], 
        #['swin_v2_b', 1024],
        #['inception_v3',0]
        #]

    test_log_count = 0 # Needed to display all five datasets

    if MLP_OPT: print('MLP is activated')
    # Used for grid search
    size = img_size#, 1440 ,1760]
    
    # Used for precent test
    
    for size in range(0,100,5): #!
        if size == 0: size =1
        crop_perc = size/100 #!
    
    # Used for fixed bounding_box
    #for backbone_name, no_filters in model_specs:
        study = optuna.create_study(direction="maximize")#, pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=N_TRIALS, timeout=None)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        
    #cli_main()