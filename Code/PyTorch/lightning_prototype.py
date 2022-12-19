from argparse import ArgumentParser
import os
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision.datasets.mnist import MNIST
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import loggers as pl_loggers



class LitClassifier(pl.LightningModule):
    def __init__(self, hidden_dim=128, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(img_size
 * img_size * 3, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
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


def cli_main():
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/test_setup')
    
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    # ------------
    # data
    # ------------

    # Create datasets for training & validation, download if necessary
    full_set = torchvision.datasets.ImageFolder('/pvol' + '/' + str(img_size)+ '_images/', transform= transforms.ToTensor()) #, transform=transform

    training_set, validation_set, test_set = torch.utils.data.random_split(full_set,[0.7, 0.2, 0.1], generator=torch.Generator().manual_seed(42))
    
    # Create data loaders for our datasets; shuffle for training and for validation
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count())
    
    print('Number of training images: {}'.format(len(train_loader.dataset)))
    
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())
    
    print('Number of validation images: {}'.format(len(val_loader.dataset)))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())
    print('Number of test images: {}'.format(len(test_loader.dataset)))


    
    # ------------
    # model
    # ------------
    model = LitClassifier(args.hidden_dim, args.learning_rate)

    # ------------
    # training
    # ------------
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    if torch.cuda.is_available(): 
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=2, logger=tb_logger)
    else:
        trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(dataloaders=test_loader)
    #model_scripted = torch.jit.script(model) # Export to TorchScript
    #print('Model is being saved!')
    #model_scripted.s