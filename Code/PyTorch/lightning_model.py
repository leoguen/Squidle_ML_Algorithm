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

class KelpClassifier(pl.LightningModule):
    def __init__(self, hidden_dim=128, learning_rate=1e-3):
        super().__init__()
        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 2
        self.classifier = nn.Linear(num_filters, num_target_classes)
        self.save_hyperparameters()

        '''
        self.l1 = torch.nn.Linear(img_size
 * img_size * 3, self.hparams.hidden_dim)
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
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = KelpClassifier.add_model_specific_args(parser)
    args = parser.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    # ------------
    # data
    # ------------

    # Create datasets for training & validation, download if necessary
    full_set = torchvision.datasets.ImageFolder('/pvol' + '/' + str(img_size)+ '_images/', transform= transforms.ToTensor()) #, transform=transform

    training_set, validation_set, test_set = torch.utils.data.random_split(full_set,[0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))
    
    # Create data loaders for our datasets; shuffle for training and for validation
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count())
    print('Dataset image size: {}'.format(img_size))
    print('Number of training images: {}'.format(len(train_loader.dataset)))
    
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())
    
    print('Number of validation images: {}'.format(len(val_loader.dataset)))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())
    print('Number of test images: {}'.format(len(test_loader.dataset)))
    
    # ------------
    # model
    # ------------
    model = KelpClassifier(args.hidden_dim, args.learning_rate)

    # ------------
    # training
    # ------------
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="/pvol/logs/lightning_logs", name=str(img_size)+'_Image_Size')
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
        trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(ckpt_path='best', dataloaders=test_loader)

    # Should work with pytorch-lightning >= 1.8.4
    #model_scripted = torch.jit.script(model)
    #print('Model is being saved!')
    #model_scripted.save("pvol/Trials/{}_trial.pth".format('test')) # Save

if __name__ == '__main__':
    img_list = [16, 24, 32, 64, 128, 224, 240, 256, 272, 288, 304, 320, 336, 512]
    for img_size in img_list:
        cli_main()
