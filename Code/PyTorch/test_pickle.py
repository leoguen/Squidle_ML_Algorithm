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
# Get list of all images in a folder
onlyfiles = [f for f in listdir(HERE + '/Ecklonia_Mix') if isfile(join(HERE + '/Ecklonia_Mix', f))]

# Load model that is saved in pickle
loaded_model = torch.jit.load(HERE + '/models/model_20221123_173129_29_36.pth')
loaded_model.eval()

# Used to classify success of model
truePos = 0
trueNeg = 0
falsePos = 0
falseNeg = 0


# Iterate through list of filenames and make predictions for each
for filename in onlyfiles:
    transform = transforms.ToTensor()
    transformed_image = transform(Image.open(HERE + '/Ecklonia_Mix/' + filename))

    # Make prediction based on loaded model
    predictions = loaded_model(transformed_image)
    
    # Use softmax to level data
    sm = torch.nn.Softmax()
    probabilities = sm(predictions) 

    # Ectract top class and probability for the class
    top_p, top_class = torch.topk(probabilities, 1)

    if top_class == 1: 
        print('Not Ecklonia Radiata with ' + str(int(top_p*100)) + ' percent certainty')
        if filename[0:1] == 'E':
            falseNeg +=1
        elif filename[0:1] == 'O':
            trueNeg += 1
        else: print('Weird file name.')
    elif top_class == 0:
        print('Ecklonia Radiata with ' + str(int(top_p*100)) + ' percent certainty')
        if filename[0:1] == 'E':
            truePos +=1
        elif filename[0:1] == 'O':
            falsePos += 1
        else: print('Weird file name.')
    else: 
        print('The code does not work properly')

print('No. of True Positives: {} \n No. of True Negatives: {} \n No. of False Positives: {} \n No. of False Negatives: {} \n'.format(truePos, trueNeg, falsePos, falseNeg))

