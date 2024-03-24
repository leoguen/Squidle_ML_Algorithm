import torch
import torch.nn as nn
from torch.nn import functional as F
import torchmetrics
import torchvision.models as models
from pytorch_lightning import LightningModule

class KelpClassifier(LightningModule):
    def __init__(self, backbone_name: str = "inception_v3", no_filters: int = 0, optimizer_name: str = "AdamW"):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.backbone_name = backbone_name
        backbone = getattr(models, backbone_name)(weights='DEFAULT')
        
        #implementing inception_v3
        if self.backbone_name == 'inception_v3': # Initialization for Inception_v3
        #self.model = models.inception_v3(weights='DEFAULT') 
            self.model = backbone
            self.model.aux_logits = False
            self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 2)
            )
        else: # Initialization for all other models
            num_filters = no_filters      
            if num_filters == 0:
                num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
            self.feature_extractor = nn.Sequential(*layers)
            num_target_classes = 2
            self.classifier = nn.Linear(num_filters,  num_target_classes)
    
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
    
    def predict_step(self, batch, batch_idx): 
        # This can be used for implementation
        x, y = batch
        #x, y = batch
        y_hat = self(x)
        #loss = F.cross_entropy(y_hat, y)
        prob = F.softmax(y_hat, dim=1)
        top_p, top_class = prob.topk(1, dim = 1)
        
        #return int(top_class.data[0][0]), float(top_p.data[0][0])
        return batch_idx, int(y[0]), int(top_class.data[0][0]), float(top_p.data[0][0])
