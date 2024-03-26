# PyTorch related
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchmetrics
import torchvision
import torchvision.models as models
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset
from torchmetrics.classification import BinaryF1Score
from pytorch_lightning import LightningModule

# Everything else
import cv2
import optuna
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
    def __init__(self, test_list, test, inception, csv_data_path, csv_file_df, img_path):
        self.test_indicator = test
        self.csv_data_path = csv_data_path
        self.img_path = img_path
        self.inception = inception
        self.csv_file_df =csv_file_df
        self.class_map = {one_word_label : 1, "Others": 0}
        compare_dir_csv(self, csv_data_path, img_path)
        # Add unpadded and padded entries to data

    def __len__(self):
        #print(self.csv_file_df.shape[0])
        return self.csv_file_df.shape[0]    
    
    def __getitem__(self, idx):
        img_path = self.img_path +self.csv_file_df.file_name.iloc[idx]
        class_id = torch.tensor(0)
        # make everything a string in the col
        self.csv_file_df[col_name] = self.csv_file_df[col_name].astype(str)
        if 'lineage' in col_name:
            if label_name in self.csv_file_df[col_name].iloc[idx]: 
                class_id = torch.tensor(1)
        else: #If translated labels are used
            if self.csv_file_df[col_name].iloc[idx] == label_name: 
                class_id = torch.tensor(1) 
        img = cv2.imread(img_path)
        #class_id = self.class_map[class_name]
        img = Image.fromarray(img)
        x = self.csv_file_df.point_x.iloc[idx] # At this point only float
        y = self.csv_file_df.point_y.iloc[idx] # At this point only float
        #print(img.size)
        x_img, y_img = img.size
        loc_img_size = img_size
        if crop_perc != 0: #Added +2 here to make up for centercrop as part of
            loc_img_size = int((x_img + y_img)/2 * (crop_perc+0.02))
        x = x_img*x #Center position
        y = y_img*y #Center position
        
        # get crop coordinates
        x0, x1, y0, y1 = get_crop_points(self, x, y, img, loc_img_size)
        cropped_img = img.crop((x0, y0, x1, y1))
        x_crop_size, y_crop_size = cropped_img.size

        if self.test_indicator: 
            train_transforms = transforms.Compose([
            transforms.CenterCrop([int(y_crop_size*0.9), int(x_crop_size*0.9)]),
            transforms.Resize((299, 299)),
            transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            ])
        else:

            train_transforms = transforms.Compose([])
            
            if RandomEqualize:
                train_transforms.transforms.append(transforms.RandomEqualize(p=0.1))
            
            train_transforms.transforms.append(transforms.ToTensor())# ToTensor : [0, 255] -> [0, 1]

            if RandomRotation:
                train_transforms.transforms.append(transforms.RandomRotation(degrees=(-20, 20)))
            if RandomErasing:
                train_transforms.transforms.append(transforms.RandomErasing(p=0.1))
            if RandomPerspective:
                train_transforms.transforms.append(transforms.RandomPerspective(p=0.1, distortion_scale=0.3))
            if RandomAffine:
                train_transforms.transforms.append(transforms.RandomAffine( degrees=(-20, 20), translate=(0.1, 0.15), scale=(0.9, 1.2)))
            if RandomVerticalFlip:
                train_transforms.transforms.append(transforms.RandomVerticalFlip(p=0.1))
            if RandomHorizontalFlip:
                train_transforms.transforms.append(transforms.RandomHorizontalFlip(p=0.1))
            if RandomInvert:
                train_transforms.transforms.append(transforms.RandomInvert(p=0.1))
            if ColorJitter:
                train_transforms.transforms.append(transforms.ColorJitter(brightness=0.2, hue=0.0, saturation=0.2, contrast=0.2))
            if ElasticTransform:
                train_transforms.transforms.append(transforms.ElasticTransform(alpha=120.0))
            if RandomAutocontrast:
                train_transforms.transforms.append(transforms.RandomAutocontrast(p=0.1))
            if RandomGrayscale:
                train_transforms.transforms.append(transforms.RandomGrayscale(p=0.1))
                
            train_transforms.transforms.append(transforms.CenterCrop([int(y_crop_size*0.9), int(x_crop_size*0.9)]))
            train_transforms.transforms.append(transforms.Resize((299, 299)))
            train_transforms.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        img_tensor = train_transforms(cropped_img)

        return img_tensor, class_id
    
class KelpClassifier(pl.LightningModule):
    def __init__(self, backbone_name, no_filters, LEARNING_RATE ,trainer, trial, img_size, batch_size, acc_val): #dropout, learning_rate, 
        super().__init__()
        # init a pretrained resnet
        self.csv_test_results = [[],[],[],[],[],[],[],[]]
        self.img_size = img_size
        self.trainer = trainer
        self.pl_module = LightningModule
        self._trial = trial
        self.batch_size = batch_size
        self.monitor = 'f1_score'
        self.hparams.learning_rate = LEARNING_RATE
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.backbone_name = backbone_name
        self.acc_val = acc_val
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
        train_dataloader = DataLoader(training_set, batch_size=self.batch_size, num_workers=30)#os.cpu_count(),shuffle=False)
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(validation_set, batch_size=self.batch_size, num_workers=30)#os.cpu_count(),shuffle=False)
        return val_dataloader
    
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
                shutil.rmtree(self.logger.log_dir, ignore_errors=True)
                raise optuna.TrialPruned(message) #!

    def forward(self, x):
        if self.backbone_name == 'inception_v3':
            self.model.eval()
            x = self.model(x)
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)
            x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        metrics, loss, f1_score, top_eck_p,top_class, res_y, prob = analyze_pred(self,x, y, self.acc_val)
        for i, metric in enumerate(metrics):
            self.training_losses[i].append(metric)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        metrics, loss, f1_score, top_eck_p, top_class, res_y, prob = analyze_pred(self,x, y, self.acc_val)
        for i, metric in enumerate(metrics):
            self.valid_losses[i].append(metric)
        # only log when not sanity checking
        if not(self.trainer.sanity_checking):
            self.log('f1_score', f1_score, on_step=False, on_epoch=True)
        #return f1_score
        image_to_tb(self, batch, batch_idx, 'train')
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        metrics, loss, f1_score,top_eck_p, top_class, res_y, prob = analyze_pred(self,x, y, self.acc_val)
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
        self.csv_test_results[path_label].extend(test_results)

    
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
        x, y = batch
        y_hat = self(x)
        #loss = F.cross_entropy(y_hat, y)
        prob = F.softmax(y_hat, dim=1)
        top_p, top_class = prob.topk(1, dim = 1)

        return batch_idx, int(y[0]), int(top_class.data[0][0]), float(top_p.data[0][0])

    def configure_optimizers(self):
        return getattr(torch.optim, optimizer_name)(self.parameters(),weight_decay=l2_param, lr=self.hparams.learning_rate)

def save_test_csv(self):
    global path_label
    
    path_list = [
    'WA', 
    'NSW_Broughton', 
    'VIC_Prom',
    'VIC_Discoverybay', 
    'TAS_Lanterns',
    'Port_Phillip_Heads',
    'Jervis_Bay',
    'Gulf_St_Vincent'
    ]
    
    if crop_perc != 0:
        dir = logger_path+log_name+'/perc_'+str(int(crop_perc*100))+'/' 
    else: 
        dir = logger_path+log_name+'/'+str(img_size)+'/'

    test_csv_path = dir + 'lightning_logs/version_{}/'.format(self.trainer.logger.version) + 'test_results_'+ path_list[path_label]+'.csv'

    df = pd.DataFrame(self.csv_test_results[path_label], columns=('Others', 'Ecklonia', 'Truth', 'Pred')) 
    
    # saving the dataframe 
    #df.to_csv(logger_path + log_name +'/'+ str(img_size)+'/lightnings_logs/test_results_'+ path_list[path_label]+'.csv')
    df.to_csv(test_csv_path)
    

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

def compare_dir_csv(self, csv_path, img_path):
    # Suppress Warning
    pd.set_option('mode.chained_assignment', None)

    #csv_path = csv_path.replace(r'(.*)/.*', '\\1', regex=True).astype('str')
    # delete every kind of ending that corresponds to jpg in the web address
    self.csv_file_df.columns = self.csv_file_df.columns.str.replace('[.]', '_', regex=True)

    #self.csv_file_df['file_name'] = self.csv_file_df.point_media_path_best.str.replace(r'(.*)\.(?i)jpg.?', '\\1', regex=True).astype('str')
    self.csv_file_df['file_name'] = self.csv_file_df.point_media_path_best.str.replace(r'(?i)(.*)\.jpg\.?', '\\1', regex=True).astype('str')

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
    print("Amount of existing images in Folder (True) and Missing (False):")
    print(self.csv_file_df.exists.value_counts())
    # Delete all entries that are not downloaded from CSV file
    self.csv_file_df = self.csv_file_df[self.csv_file_df.exists]

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

def analyze_pred(self,x,y, acc_val):
        y_hat = self(x)
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
        if acc_val == "gpu":
            f1_metric = BinaryF1Score().to('cuda')
        else:
            f1_metric = BinaryF1Score()
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

    parser.add_argument('--limit_train_batches', metavar='ltrb', type=float, help='Limits the amount of entries in the trainer for debugging purposes', default=0.01) #!

    parser.add_argument('--limit_val_batches', metavar='lvb', type=float, help='Limits the amount of entries in the trainer for debugging purposes', default=0.3) #!

    parser.add_argument('--limit_test_batches', metavar='lteb', type=float, help='Limits the amount of entries in the trainer for debugging purposes', default=0.1) #!

    parser.add_argument('--epochs', metavar='epochs', type=int, help='The number of epochs the algorithm trains', default=1) #!

    parser.add_argument('--log_path', metavar='log_path', type=str, help='The path where the logger files are saved', default='./Logs/')

    parser.add_argument('--log_name', metavar='log_name', type=str, help='Name of the experiment.', default='unnamed')

    parser.add_argument('--img_path', metavar='img_path', type=str, help='Path to the database of images', default='./Images/') #/pvol/Seagrass_Database/

    parser.add_argument('--csv_path', metavar='csv_path', type=str, help='Path to the csv file describing the images', default='./Annotationsets/29219_neighbour_Sand _ mud (_2mm).csv')
    #/pvol/Final_Eck_1_to_10_Database/Original_images/22754_neighbour_Seagrass_cover_NMSC_list.csv
    #/pvol/Final_Eck_1_to_10_Database/Original_images/205282_neighbour_Hard_coral_cover_NMSC_list.csv
    #/pvol/Final_Eck_1_to_10_Database/Original_images/405405_neighbour_Macroalgal_canopy_cover_NMSC_list.csv
    #/pvol/Final_Eck_1_to_10_Database/Original_images/164161_1_to_1_neighbour_Ecklonia_radiata_except.csv
    #/pvol/Final_Eck_1_to_1_neighbour_Database/Original_images/164161_1_to_1_neighbour_Ecklonia_radiata_except.csv

    parser.add_argument('--n_trials', metavar='n_trials', type=int, help='Number of trials that Optuna runs for', default=1) #!

    parser.add_argument('--backbone', metavar='backbone', type=str, help='Name of the model which should be used for transfer learning', default='inception_v3')

    parser.add_argument('--real_test',  help='If True: a seperate dataset is used, if False dataset is extracted out of training set. ', action='store_false') #!

    parser.add_argument('--test_img_path', metavar='test_img_path', type=str, help='Path to the database of test images', default='/pvol/Ecklonia_Testbase/NSW_Broughton/')

    parser.add_argument('--img_size', metavar='img_size', type=int, help='Defines the size of the used image.', default=299)
    
    parser.add_argument('--crop_perc', metavar='crop_perc', type=float, help= 'Defines percentage that is used to crop the image.', default=0.16)

    parser.add_argument('--batch_size', metavar='batch_size', type=int, help= 'Defines batch_size that is used to train algorithm.', default=192) #!

    parser.add_argument('--label_name', metavar='label_name', type=str, help='Name of the label used in the csv file', default='Physical > Substrate > Unconsolidated (soft) > Sand / mud (<2mm)') #Seagrass cover
    #Macroalgal canopy cover

    parser.add_argument('--cross_validation', metavar='cross_validation', type=int, help= 'Defines how many sets the dataset is going to be divided in for cross validation.', default=0) #!
    
    parser.add_argument('--col_name', metavar='col_name', type=str, help='Name of the column that should be used for filtering e.g. "label_name", "translated_label_name", "lineage_name"', default='label_translated_lineage_names') 
    
    parser.add_argument('--grid_search',  help='If True: a grid search from 1 to 99 percent in 5 percent increment.', action='store_true') #!

    args =parser.parse_args()
    no_filters = 0
    return args.percent_valid_examples,args.percent_test_examples,args.limit_train_batches,args.limit_val_batches,args.limit_test_batches,args.epochs,args.log_path,args.log_name,args.img_path, args.n_trials,args.backbone, no_filters, args.real_test, args.test_img_path, args.img_size, args.csv_path, args.crop_perc, args.batch_size, args.label_name, args.cross_validation, args.col_name, args.grid_search

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

def display_dataloader(data_loader, name):
    counter = [0,0]
    #Used for CSV Dataset
    
    for idx in data_loader.dataset.indices:
        if 'lineage' in col_name: # in case different depths of lineage are required
            if label_name in data_loader.dataset.dataset.csv_file_df[col_name].iloc[idx]:
                #print('Eck Label ' + data_loader.dataset.dataset.csv_file_df.label_name[idx])
                counter[1] += 1 
            else: 
                #print('Others label '+ (data_loader.dataset.dataset.csv_file_df.label_name[idx]))
                counter[0] += 1
        else: #Used if not lineage
            if data_loader.dataset.dataset.csv_file_df[col_name].iloc[idx] == label_name:
                #print('Eck Label ' + data_loader.dataset.dataset.csv_file_df.label_name[idx])
                counter[1] += 1 
            else: 
                #print('Others label '+ (data_loader.dataset.dataset.csv_file_df.label_name[idx]))
                counter[0] += 1
    print('The {} comprises of: {} {} & Others {} \n'.format(name, label_name, counter[1], counter[0]))
    

def get_test_sets(csv_file_df):
    # Group by campaign and count total entries and entries matching the label
    result = csv_file_df.groupby('point_media_deployment_campaign_key').agg(
        total_entries=pd.NamedAgg(column=col_name, aggfunc='size'),
        matching_entries=pd.NamedAgg(column=col_name, aggfunc=lambda x: (x == label_name).sum())
    )
    
    # Calculate the ratio of matching_entries to total_entries
    result['ratio'] = result['matching_entries'] / result['total_entries']
    
    # Filter datasets where the ratio is more than 20% and total_entries is more than 1000
    filtered_result = result[(result['ratio'] > 0.2) & (result['total_entries'] >= 1000)]
    
    # Sort the filtered result DataFrame based on total_entries
    sorted_result = filtered_result.sort_values(by='total_entries')
    
    # Get the campaign keys for the three smallest datasets (if they exist)
    campaigns_to_extract = sorted_result.head(3).index.tolist()
    
    # Initialize empty DataFrames for the test sets
    df1, df2, df3 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    if len(campaigns_to_extract) > 0:
        df1 = csv_file_df[csv_file_df['point_media_deployment_campaign_key'] == campaigns_to_extract[0]]
        print(f"Length of first test dataset: {len(df1)}")
    if len(campaigns_to_extract) > 1:
        df2 = csv_file_df[csv_file_df['point_media_deployment_campaign_key'] == campaigns_to_extract[1]]
        print(f"Length of second test dataset: {len(df2)}")
    if len(campaigns_to_extract) > 2:
        df3 = csv_file_df[csv_file_df['point_media_deployment_campaign_key'] == campaigns_to_extract[2]]
        print(f"Length of third test dataset: {len(df3)}")
    
    # Remove these rows from the original DataFrame
    csv_file_df = csv_file_df[~csv_file_df['point_media_deployment_campaign_key'].isin(campaigns_to_extract)]
    
    return df1, df2, df3, csv_file_df



def objective(trial: optuna.trial.Trial) -> float:

    ###############
    # Optuna Params
    ###############
    global RandomEqualize, RandomRotation, RandomErasing, RandomPerspective, RandomAffine, RandomVerticalFlip, RandomHorizontalFlip, RandomInvert, ColorJitter, ElasticTransform, RandomAutocontrast, RandomGrayscale

    RandomEqualize, RandomRotation, RandomVerticalFlip, RandomHorizontalFlip  = True, True, True, True
    
    RandomInvert, RandomAutocontrast, RandomErasing, RandomPerspective, RandomAffine, ColorJitter, ElasticTransform, RandomGrayscale = False, False, False, False, False, False, False, False
    
    LEARNING_RATE = 1e-5

    global threshold
    threshold = 0.5

    ##############
    # Data Loading
    ##############
    inception = False
    if backbone_name == 'inception_v3': 
        inception = True

    test_list = []

    csv_file_df= pd.read_csv(csv_path, on_bad_lines='skip', low_memory=False) 

    test_df1, test_df2, test_df3, train_val_set = get_test_sets(csv_file_df)

    #train_val_set = GeneralDataset(img_size, test_list, test = False, inception = inception, test_img_path = csv_path)
    train_val_set = CSV_Dataset(test_list, test = False, inception = inception, csv_data_path = csv_path, csv_file_df=train_val_set, img_path=img_path)
    
    global training_set, validation_set
    training_set, validation_set = torch.utils.data.random_split(train_val_set,[0.80, 0.20], generator=torch.Generator().manual_seed(4234))
    
    if cross_validation != 0:
        train_index = [0,1,2,3,4]
        train_index.remove(cross_counter)
        cross_sets = []*cross_validation
        cross_sets = torch.utils.data.random_split(train_val_set,[0.2,0.2,0.2,0.2,0.2], generator=torch.Generator().manual_seed(4234))
        validation_set = cross_sets[cross_counter]

        training_set = torch.utils.data.ConcatDataset([cross_sets[train_index[0]],cross_sets[train_index[1]], cross_sets[train_index[2]], cross_sets[train_index[3]] ])

    acc_val = 'cpu'
    if torch.cuda.is_available(): acc_val = 'gpu'
    
    if crop_perc != 0:
        dir = logger_path+log_name+'/perc_'+str(int(crop_perc*100))+'/' 
    else: 
        dir = logger_path+log_name+'/'+str(img_size)+'/'

    trainer = pl.Trainer(
        logger=True,
        default_root_dir=dir,
        enable_checkpointing=True,
        max_epochs=epochs,
        accelerator=acc_val,
        #callbacks=[EarlyStopping(monitor="f1_score", mode="max")],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        precision=16,
        log_every_n_steps=50,
        #auto_scale_batch_size="binsearch",
        #auto_lr_find=False, #!
    )
    
    model = KelpClassifier(backbone_name, 
                            no_filters, 
                            LEARNING_RATE, 
                            batch_size = batch_size, 
                            trial = trial, 
                            #monitor ='f1_score', 
                            trainer= trainer,  
                            #pl_module = LightningModule, 
                            img_size=img_size,
                            acc_val=acc_val)
    #trainer.tune(model)
    trainer.fit(model)
    

    ##########
    # Logger #
    ##########
    hyperparameters = dict(
            optimizer_name = optimizer_name,
            learning_rate=LEARNING_RATE,#model.hparams.learning_rate,  
            backbone = backbone_name,
            #batchsize = model.batch_size, 
            threshold = threshold,
            f1_score = trainer.callback_metrics["f1_score"].item(),
            RandomEqualize=RandomEqualize, 
            RandomRotation=RandomRotation, 
            RandomErasing=RandomErasing, 
            RandomPerspective=RandomPerspective, 
            RandomAffine=RandomAffine, 
            RandomVerticalFlip=RandomVerticalFlip, 
            RandomHorizontalFlip=RandomHorizontalFlip, 
            RandomInvert=RandomInvert, 
            ColorJitter=ColorJitter, 
            ElasticTransform=ElasticTransform, 
            RandomAutocontrast=RandomAutocontrast, RandomGrayscale=RandomGrayscale
            )
    if crop_perc != 0:
        hyperparameters['crop_perc']=crop_perc
    else:
        hyperparameters['image_size']= img_size 
    if not(real_test):
        hyperparameters['test_perc'] = percent_test_examples 


    trainer.logger.log_hyperparams(hyperparameters)

    
    # Handle pruning based on the intermediate value.
    if trial.should_prune(): #!
        raise optuna.TrialPruned()
    
    # value copied here so that it is not influenced by different testing options
    f1_score_end = trainer.callback_metrics["f1_score"].item()

    #Used for Ecklonia
    path_list = [
        '/pvol/Ecklonia_Testbase/WA/Original_images/annotations-u45-leo_kelp_SWC_WA_AI_test-leo_kelp_AI_SWC_WA_test_25pts-8148-7652a9b48f0e3186fe5d-dataframe.csv', 
        '/pvol/Ecklonia_Testbase/NSW_Broughton/Original_images/annotations-u45-leo_kelp_AI_test_broughton_is_NSW-leo_kelp_AI_test_broughton_is_25pts-8152-7652a9b48f0e3186fe5d-dataframe.csv', 
        '/pvol/Ecklonia_Testbase/VIC_Prom/Original_images/annotations-u45-leo_kelp_AI_test_prom_VIC-leo_kelp_AI_test_prom_25pts-8150-7652a9b48f0e3186fe5d-dataframe.csv',
        '/pvol/Ecklonia_Testbase/VIC_Discoverybay/Original_images/annotations-u45-leo_kelp_AI_test_discoverybay_VIC_phylospora-leo_kelp_AI_test_db_phylospora_25pts-8149-7652a9b48f0e3186fe5d-dataframe.csv', 
        '/pvol/Ecklonia_Testbase/TAS_Lanterns/Original_images/annotations-u45-leo_kelp_AI_test_lanterns_TAS-leo_kelp_AI_test_lanterns_25pts-8151-7652a9b48f0e3186fe5d-dataframe.csv', 
        '/pvol/Ecklonia_Testbase/Gulf_St_Vincent/Original_images/Ecklonia_RLS_Gulf St Vincent_2012.csv',
        '/pvol/Ecklonia_Testbase/Jervis_Bay/Original_images/Ecklonia_RLS_Jervis Bay Marine Park_2015.csv',
        '/pvol/Ecklonia_Testbase/Port_Phillip_Heads/Original_images/Ecklonia_RLS_Port Phillip Heads_2010.csv'
        ]
    
    # Used for Hardcoral
    path_list = [
        '/pvol/NMSC_Testbase/Hardcoral_Queensland/Original_images/Hardcoral_RLS_Queensland (other)_2015.csv',
        '/pvol/NMSC_Testbase/Hardcoral_Norfolk/Original_images/Hardcoral_RLS_Norfolk Island_2013.csv',
        '/pvol/NMSC_Testbase/Hardcoral_Ningaloo/Original_images/Hardcoral_RLS_Ningaloo Marine Park_2016.csv'
    ]

    # Used for Macroalgal
    path_list = [
        '/pvol/NMSC_Testbase/Macroalgal_Port_Phillip/Original_images/Macroalgal_RLS_Port Phillip Bay_2017.csv',
        '/pvol/NMSC_Testbase/Macroalgal_Gulf/Original_images/Macroalgal_RLS_Gulf St Vincent_2012.csv',
        '/pvol/NMSC_Testbase/Macroalgal_Jervis/Original_images/Macroalgal_RLS_Jervis Bay Marine Park_2015.csv'
    ]

    # Used for Seagrass
    path_list = [
        '/pvol/NMSC_Testbase/Seagrass_Port_Phillip_2010/Original_images/Seagrass_RLS_Port Phillip Heads_2010.csv',
        '/pvol/NMSC_Testbase/Seagrass_Port_Phillip_2016/Original_images/Seagrass_RLS_Port Phillip Bay_2016.csv',
        '/pvol/NMSC_Testbase/Seagrass_Port_Phillip_2017/Original_images/Seagrass_RLS_Port Phillip Bay_2017.csv'
    ]
    # Testset loading
    for idx ,test_set in enumerate([test_df1, test_df2, test_df3]):
        global path_label 
        path_label = idx
        test_set = CSV_Dataset(test_list, test = True, inception=inception, csv_data_path=csv_path, csv_file_df=test_set, img_path=img_path)
        # Just looks at one dataset
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=30)#os.cpu_count())

        trainer.test(ckpt_path='best', dataloaders=test_loader)
    
    print('Number of samples overall: {}'.format(len(train_val_set) + len(test_set)))
    
    return f1_score_end

if __name__ == '__main__':
    percent_valid_examples, percent_test_examples, limit_train_batches, limit_val_batches, limit_test_batches, epochs, logger_path, log_name, img_path, n_trials, backbone_name, no_filters, real_test, test_img_path, img_size, csv_path, crop_perc, batch_size, label_name, cross_validation, col_name, grid_search = get_args()

    one_word_label = label_name.split()[0] #only use first word
    cross_counter = 0
    # Used for fixed bounding_box
    backbone_name, no_filters ='inception_v3', 0
    optimizer_name = 'AdamW'
    l2_param = 0.01

    
    #for cross_counter in range(cross_validation):
        
        
        # Used for precent test
    #for size in range(12,28,2): #!
    #for size in [1,5,8,10,12,14,15,16,18,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]:
    #for size in [10,12,14,15,16,18,20,25,30,35,40]:
    
    #crop_perc = size/100
        #if size == 0: size =1
    
    if grid_search == True: 
        for_array = range(1,99,5)
        for_array = [float(num) / 100 for num in for_array]
    elif cross_validation != 0:
        for_array = range(cross_validation)
    else: 
        for_array = [1]

    for i in for_array:
        # If this is a Gridsearch update the crop_perc for each iteration
        if grid_search:
            crop_perc = i
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials, timeout=None)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

                #importance_dict = optuna.importance.get_param_importances(study)
                #print(importance_dict)
                        
                    #cli_main()