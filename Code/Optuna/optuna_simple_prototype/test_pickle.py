import pickle
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import os
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join


HERE = os.path.dirname(os.path.abspath(__file__))

# Get list of all images in a folder
onlyfiles = [f for f in listdir(HERE + '/Ecklonia_dataset') if isfile(join(HERE + '/Ecklonia_dataset', f))]

# Load model that is saved in pickle
loaded_model = pickle.load(open('Optuna/optuna_simple_prototype/Trials/Saved/best_trial_Nov-22-2022-12-29_89.pickle', 'rb'))
loaded_model.eval()

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
    print(filename, predictions)
    lst_prob = prob.tolist()
    
    '''
    if top_class == 0: 
        print('Not Ecklonia Radiata with ' + str(float(top_p[0]*100)) + ' percent certainty')
    elif top_class[0] == 1:
        print('Ecklonia Radiata with ' + str(float(top_p[0]*100)) + ' percent certainty')
    else: 
        print('The code does not work properly')
    '''