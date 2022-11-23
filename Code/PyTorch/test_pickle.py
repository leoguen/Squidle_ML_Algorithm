import pickle
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join


HERE = os.path.dirname(os.path.abspath(__file__))
no_ecklonia = 0
no_others = 0
# Get list of all images in a folder
onlyfiles = [f for f in listdir(HERE + '/Ecklonia_Mix') if isfile(join(HERE + '/Ecklonia_Mix', f))]

# Load model that is saved in pickle
loaded_model = torch.jit.load(HERE + '/models/model_20221123_173129_29_36.pth')
loaded_model.eval()

# Iterate through list of filenames and make predictions for each
for filename in onlyfiles:
    transform = transforms.ToTensor()
    transformed_image = transform(Image.open(HERE + '/Ecklonia_Mix/' + filename))

    # Make prediction based on 
    predictions = loaded_model(transformed_image)
    labels = torch.argmax(predictions, 1)
    #print(predictions)
    #_, prob = torch.max(predictions, dim=1)
    prob = nnf.softmax(predictions, dim=0)
    top_p, top_class = prob.topk(1, dim = 1)
    print(filename, predictions, labels)
    lst_prob = prob.tolist()
    
    if labels[0] == 0: 
        print('Not Ecklonia Radiata with ' + str(float(top_p[0]*100)) + ' percent certainty')
        no_others +=1 
    elif labels[0] == 1:
        print('Ecklonia Radiata with ' + str(float(top_p[0]*100)) + ' percent certainty')
        no_ecklonia +=1
    else: 
        print('The code does not work properly')

    # !!! WHY DOES TOP_P AND TOP_CLASS NOT WORK

    print('The current ratio is Ecklonia: ' + str(no_ecklonia) + ' to Others ' + str(no_others))

