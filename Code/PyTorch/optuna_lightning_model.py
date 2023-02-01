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


writer = SummaryWriter()

PERCENT_VALID_EXAMPLES = 0.1
PERCENT_TEST_EXAMPLES = 0.1
CLASSES = 2
EPOCHS = 50
LOGGER_PATH = '/pvol/logs/'
LOG_NAME = 'Adv_Log_Imagenet_Mean/'
IMG_PATH = '/pvol/Ecklonia_Database/'

class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.
    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        '''
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return
        '''
        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)

class GeneralDataset(Dataset):
    def __init__(self, img_size, test_list, test):
        if test: # True test dataset is returned
            # Add unpadded and padded entries to data
            self.data = test_list[0]
            self.data = self.data + test_list[1]
            self.class_map = {"Ecklonia" : 0, "Others": 1}
        else: 
            self.imgs_path = IMG_PATH + str(img_size)+ '_images/'
            file_list = [self.imgs_path + 'Others', self.imgs_path + 'Ecklonia']
            self.data = []
            for class_path in file_list:
                class_name = class_path.split("/")[-1]
                for img_path in glob.glob(class_path + "/*.jpg"):
                    self.data.append([img_path, class_name])
            
            # Delete all entries that are used in test_list
            del_counter = len(self.data)
            #print('Loaded created dataset of {} entries. \nNow deleting {} duplicate entries.'.format(len(self.data), len(test_list[0])))
            for test_entry in (test_list[0]):
                if test_entry in self.data:
                    self.data.remove(test_entry)
                    # !!!Why does it only remove half of all duplicate entries?
            #print('Deleted {} duplicate entries'.format(del_counter-len(self.data)))
            self.class_map = {"Ecklonia" : 0, "Others": 1}
        
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        class_id = self.class_map[class_name]
        train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        img_tensor = train_transforms(img)
        class_id = torch.tensor(class_id)
        return img_tensor, class_id

class UniformDataset(Dataset):
    def __init__(self, img_size):
        self.imgs_path = IMG_PATH + str(img_size)+ '_images/'
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
        train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
        )
        img_tensor = train_transforms(img)
        class_id = torch.tensor(class_id)
        return img_tensor, class_id

class MixedDataset(Dataset):
    def __init__(self, img_size, PERCENT_TEST_EXAMPLES):
        self.data = []
        self.imgs_path = IMG_PATH + str(img_size)+ '_images/'
        self.pad_imgs_path = IMG_PATH + str(img_size)+ '_images/Padding/'
        #####################
        # Get unpadded images
        #####################
        file_list = [self.imgs_path + 'Others', self.imgs_path + 'Ecklonia']
        if PERCENT_TEST_EXAMPLES*0.04*2*100 < 1:eck_perc = 1
        else: eck_perc = PERCENT_TEST_EXAMPLES*0.04*2*100
        unpad_perc = [PERCENT_TEST_EXAMPLES*100, eck_perc]
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
        #print('Others: {}, Ecklonia: {}'.format(oth_count, eck_count))
        #####################
        # Get padded images
        #####################
        file_list = [self.pad_imgs_path + 'Others', self.pad_imgs_path + 'Ecklonia']

        oth_pad_files = (len([name for name in os.listdir(file_list[0]) if os.path.isfile(os.path.join(file_list[0], name))])) 
        eck_pad_files =(len([name for name in os.listdir(file_list[1]) if os.path.isfile(os.path.join(file_list[1], name))]))

        len_both = (oth_pad_files+eck_pad_files)
        adap_len_both = len_both*PERCENT_TEST_EXAMPLES
        oth_perc = int(PERCENT_TEST_EXAMPLES*100)
        if int(PERCENT_TEST_EXAMPLES*0.04*100) == 0: eck_perc =1
        else: eck_perc = int(PERCENT_TEST_EXAMPLES*0.04*100)
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
        #print('The dataset comprises of: \nUniform Ecklonia {}\nUniform Others {} \nPadded Ecklonia {}\nPadded Others {}\nDataset length {}'.format(eck_count, oth_count, pad_eck_count, pad_oth_count, len(self.data) ))
    
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        class_id = self.class_map[class_name]
        train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
        )
        img_tensor = train_transforms(img)
        #img_tensor = transforms.ToTensor()(img)
        #img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
        class_id = torch.tensor(class_id)
        return img_tensor, class_id


class KelpClassifier(pl.LightningModule):
    def __init__(self, backbone_name, no_filters, learning_rate, dropout, batch_size):
        super().__init__()
        # init a pretrained resnet

        self._trial = optuna.trial.Trial
        self.monitor = 'f1_score'
        self.hparams.learning_rate = learning_rate
        self.accuracy = torchmetrics.Accuracy(task='binary')      
        backbone = getattr(models, backbone_name)(weights='DEFAULT')

        num_filters = no_filters
        if num_filters == 0:
            num_filters = backbone.fc.in_features
        
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        num_target_classes = 2
        
        self.classifier = nn.Linear(num_filters,  num_target_classes)
        self.training_losses = [[],[],[],[],[]]
        self.save_hyperparameters()
    '''
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
        
        print('Pruning algorithm runs')
        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)
    '''
    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        prob = F.softmax(y_hat, dim=1)
        loss = F.cross_entropy(y_hat, y)
        top_p, top_class = prob.topk(1, dim = 1)
        top_class = torch.reshape(top_class, (-1,))
        accuracy = self.accuracy(top_class, y)
        
        f1_metric = BinaryF1Score().to('cuda')
        f1_score = f1_metric(top_class, y)
        prec_metric = BinaryPrecision().to('cuda')
        prec_score = prec_metric(top_class, y)
        rec_metric = BinaryRecall().to('cuda')
        rec_score = rec_metric(top_class, y)
        metrics=[loss.item(), accuracy.item(), f1_score.item(), prec_score.item(), rec_score.item()]
        for i, metric in enumerate(metrics):
            self.training_losses[i].append(metric)
        return loss
    
    def on_train_epoch_end(self):
        #value, var, name
        name = 'train'
        metric = ['loss', 'accuracy', 'f1_score', 'precision', 'recall']
        for i, var in enumerate(metric):
            metric_list = self.training_losses[i]
            metric_mean = np.mean(metric_list)
            log_to_graph(self, metric_mean, var, name, self.global_step)
        self.training_losses = [[],[],[],[],[]]  # reset for next epoch
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        prob = F.softmax(y_hat, dim=1)
        loss = F.cross_entropy(y_hat, y)
        top_p, top_class = prob.topk(1, dim = 1)
        top_class = torch.reshape(top_class, (-1,))
        accuracy = self.accuracy(top_class, y)
        batch_idx = image_to_tb(self, batch, batch_idx)
        
        f1_metric = BinaryF1Score().to('cuda')
        f1_score = f1_metric(top_class, y)
        prec_metric = BinaryPrecision().to('cuda')
        prec_score = prec_metric(top_class, y)
        rec_metric = BinaryRecall().to('cuda')
        rec_score = rec_metric(top_class, y)
        log_to_graph(self, rec_score, 'recall', 'valid', self.global_step)
        log_to_graph(self, prec_score, 'precision', 'valid', self.global_step)
        log_to_graph(self, f1_score, 'f1_score', 'valid', self.global_step)
        log_to_graph(self, accuracy, 'accuracy', 'valid', self.global_step)
        log_to_graph(self, loss, 'loss', 'valid', self.global_step)
        self.log('f1_score', f1_score)
        return f1_score

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        prob = F.softmax(y_hat, dim=1)
        top_p, top_class = prob.topk(1, dim = 1)
        top_class = torch.reshape(top_class, (-1,))
        accuracy = self.accuracy(top_class, y)
        f1_metric = BinaryF1Score().to('cuda')
        f1_score = f1_metric(top_class, y)
        prec_metric = BinaryPrecision().to('cuda')
        prec_score = prec_metric(top_class, y)
        rec_metric = BinaryRecall().to('cuda')
        rec_score = rec_metric(top_class, y)
        self.log('test_recall', rec_score)
        self.log('test_precision', prec_score)
        #self.log('test_f1_score', f1_score)
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)
    
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


def log_to_graph(self, value, var, name ,global_step):
    self.logger.experiment.add_scalars(var, {name: value},global_step)

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

def get_test_dataset(img_size, PERCENT_TEST_EXAMPLES):
    unpad_path = IMG_PATH + str(img_size)+ '_images/'
    pads_path = IMG_PATH + str(img_size)+ '_images/Padding/'
    both_path = [unpad_path, pads_path]
    unpad_file_list = [unpad_path + 'Others', unpad_path + 'Ecklonia']
    pad_file_list =  [pads_path + 'Others', pads_path + 'Ecklonia']
    both_file_list = [unpad_file_list, pad_file_list]
    perc = [PERCENT_TEST_EXAMPLES * 100, 1]
    unpad_data = []
    pad_data = []
    both_data = [unpad_data, pad_data]
    perc_id = 0
    counter = [[0,0],[0,0]]
    if int(PERCENT_TEST_EXAMPLES * 0.05 * 100) ==0: perc[1] = 1
    else: perc[1] = int(PERCENT_TEST_EXAMPLES * 0.05 * 100)
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
    #print('The Test-Dataset comprises of: \nUniform Ecklonia {}\nUniform Others {} \nPadded Ecklonia {}\nPadded Others {}'.format(counter[0][1], counter[0][0], counter[1][1], counter[1][0]))
    return both_data
    
def objective(trial: optuna.trial.Trial) -> float:
    img_size = 288
    backbone_name, no_filters = ['regnet_x_32gf', 0]
    # We optimize the number of layers, hidden units in each layer and dropouts.
    
    ###############
    # Optuna Params
    ###############
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    BATCHSIZE = 64#trial.suggest_int("batchsize", 8, 128)
    LEARNING_RATE = trial.suggest_float(
        "learning_rate_init", 1e-6, 1e-4, log=True
    ) #min needs to be 1e-6
    ##############
    # Data Loading
    ##############
    test_list = get_test_dataset(img_size, PERCENT_TEST_EXAMPLES)
    test_set = GeneralDataset(img_size, test_list, test = True)
    train_val_set = GeneralDataset(img_size, test_list, test = False)
    
    training_set, validation_set = torch.utils.data.random_split(train_val_set,[0.90, 0.10], generator=torch.Generator().manual_seed(43))

    # Create data loaders for our datasets; shuffle for training and for validation
    train_loader = torch.utils.data.DataLoader(training_set, BATCHSIZE, shuffle=True, num_workers=os.cpu_count())
    
    val_loader = torch.utils.data.DataLoader(validation_set, BATCHSIZE, shuffle=False, num_workers=os.cpu_count())

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=os.cpu_count())
    model = KelpClassifier(backbone_name, no_filters, LEARNING_RATE, dropout, BATCHSIZE)

    #tb_logger = pl_loggers.TensorBoardLogger(save_dir="/lightning_logs/", name=str(img_size)+'_'+trial_name+'_Image_Size')

    acc_val = 'cpu'
    if torch.cuda.is_available(): acc_val = 'gpu'
    trainer = pl.Trainer(
        logger=True,
        default_root_dir=LOGGER_PATH+LOG_NAME,
        enable_checkpointing=True,
        max_epochs=EPOCHS,
        accelerator=acc_val,
        callbacks=[EarlyStopping(monitor="f1_score", mode="max")],
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        precision=16,
        log_every_n_steps=50
    )

    hyperparameters = dict(dropout=dropout, batchsize=BATCHSIZE, learning_rate=LEARNING_RATE)
    trainer.logger.log_hyperparams(hyperparameters)
    #trainer.logger.log('batchsize', BATCHSIZE)
    #trainer.logger.log('learning_rate', LEARNING_RATE)
    trainer.fit(model, train_loader,val_loader )
    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    #trainer.test(ckpt_path='best', dataloaders=test_loader)

    return trainer.callback_metrics["f1_score"].item()

if __name__ == '__main__':

    #pruning = True

    #pruner: optuna.pruners.BasePruner = (
    #    optuna.pruners.MedianPruner()
    #)

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner()) #pruner=pruner)
    study.optimize(objective, n_trials=None, timeout=None)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    #cli_main()