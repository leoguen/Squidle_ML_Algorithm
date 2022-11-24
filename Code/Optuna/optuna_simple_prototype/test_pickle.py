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
onlyfiles = [f for f in listdir(HERE + '/Ecklonia_dataset') if isfile(join(HERE + '/Ecklonia_dataset', f))]

# Load model that is saved in pickle
loaded_model = torch.jit.load(HERE + '/Trials/Saved/best_trial_Nov-24-2022-17-43_75.pth')
loaded_model.eval()
print(loaded_model)

# Used to classify success of model
truePos = 0
trueNeg = 0
falsePos = 0
falseNeg = 0

for filename in onlyfiles:
    transform = transforms.ToTensor()
    transformed_image = transform(Image.open(HERE + '/Ecklonia_dataset/' + filename))

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



'''
# Iterate through list of filenames and make predictions for each
for filename in onlyfiles:
    np_image = np.array(Image.open(HERE + '/Ecklonia_dataset/' + filename))
    tensor_image = torch.from_numpy(np_image)

    # Reshape sample to (batch-size, width x height) but batch-size is 1 
    tensor_image = torch.reshape(tensor_image, ( 1, 1728))
    
    # Convert to float32 as default is float64 and does not align with model
    tensor_image = tensor_image.to(torch.float32)

    # Make prediction based on 
    predictions = loaded_model(tensor_image)
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

    print('The current ratio is Ecklonia: ' + str(no_ecklonia) + ' to Others ' + str(no_others))
'''