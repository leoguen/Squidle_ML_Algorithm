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
import io
import os
import access_sq_images as sq

HERE = os.path.dirname(os.path.abspath(__file__))
annotation_path = '/home/ubuntu/IMAS/Code/PyTorch/Annotation_Sets/1195753_REVIEWED_ANNOTATION_LIST.csv'
bounding_box = [24, 24]
save_path = HERE + '/random_validation/'+str(bounding_box[0])+'_images'

# Get list of models
model_list = [f for f in listdir(HERE + '/models') if isfile(join(HERE + '/models', f))]
# Delete all models with wrong ending
model_list = [val for val in model_list if val.endswith(".pth")]
print('This is the modellist: {}'.format(model_list))

'''
# Get random patch from dataset
csv_file = sq.random_csv_except(annotation_path, 0.01)


# Create directory structure
sq.create_directory_structure(bounding_box=bounding_box, save_path=save_path)
#/home/ubuntu/IMAS/Code/PyTorch/random_validation/24_images/Ecklonia
# Download images from random csv file
sq.download_images(save_path=save_path, bounding_box=bounding_box, csv_file=csv_file)
'''

# Get list of all Ecklonia entries
eck_files = [f for f in listdir(save_path + '/Ecklonia') if isfile(join(save_path + '/Ecklonia', f))]
# Get list of all Other entries
other_files = [f for f in listdir(save_path + '/Others') if isfile(join(save_path + '/Others', f))]

both_dir = [[eck_files, save_path + '/Ecklonia'], [other_files, save_path + '/Others']]


# Used to classify success of model
truePos = 0
trueNeg = 0
falsePos = 0
falseNeg = 0
results=[[None]*6]*len(model_list)

def load_model(model_name):
    # For Model trained on CPU and executed on CPU
    loaded_model = torch.jit.load(HERE + '/models/' + model_name, map_location='cpu')
    '''
    try:
        # Load model that is saved in pth
        loaded_model = torch.jit.load(HERE + '/models/' + model_name)
    # For Model trained on GPU and executed on CPU
    except:
        loaded_model = torch.jit.load(HERE + '/models/' + model_name, map_location='cpu')
    '''
    #loaded_model = torch.jit.load(HERE + '/models/model_20221123_173129_0_68.pth')
    loaded_model.eval()
    return loaded_model


# Iterate through list of filenames and make predictions for each
for index, model_name in enumerate(model_list):
    loaded_model = load_model(model_name)
    print('Validating model: {}'.format(model_name))
    img_index = 0
    for dir_name, path in both_dir:
        for filename in dir_name:
            
            transform = transforms.ToTensor()

            transformed_image = transform(Image.open(path +'/'+ filename))
            try:
                # Reshape to fit model input
                reshaped_image = torch.reshape(transformed_image, (1, 1728))
            except:
                #print('Problem with image {}'.format(filename) )
                continue
            
            try:
                # Make prediction based on loaded model
                predictions = loaded_model(reshaped_image)
            except: # image needs to be reshaped for CNN
                print('CNN index {} out of {}.'.format(img_index, len(eck_files)+len(other_files)))
                # Reshape to fit model input
                cnn_reshaped_image = torch.reshape(transformed_image, (1, 3, 24, 24))
                # Make prediction based on loaded model
                predictions = loaded_model(cnn_reshaped_image)
            
            img_index += 1
            # Use softmax to level data
            #sm = torch.nn.Softmax()
            probabilities = (predictions) 

            # Ectract top class and probability for the class
            top_p, top_class = torch.topk(probabilities, 1)

            if top_class == 1: 
                #print('Not Ecklonia Radiata with ' + str(int(top_p*100)) + ' percent certainty')
                if filename.startswith('Ecklonia'):
                    falseNeg +=1
                elif not(filename.startswith('Ecklonia')):
                    trueNeg += 1
                else: print('Weird file name.')
            elif top_class == 0:
                #print('Ecklonia Radiata with ' + str(int(top_p*100)) + ' percent certainty')
                if filename.startswith('Ecklonia'):
                    truePos +=1
                elif not(filename.startswith('Ecklonia')):
                    falsePos += 1
                else: print('Weird file name.')
            else: 
                print('The code does not work properly')
    
    accuracy = (trueNeg+truePos)/(trueNeg+truePos+falseNeg+falsePos)
    results[index] = model_name, truePos, trueNeg, falsePos, falseNeg, accuracy
    truePos, trueNeg, falsePos, falseNeg = 0, 0, 0, 0

results = sorted(results, key=lambda x: x[5], reverse=True)
for i in range(len(model_list)):
    if i == 0:
        print(' Best model {}: \n No. of True Positives: {} \n No. of True Negatives: {} \n No. of False Positives: {} \n No. of False Negatives: {} \n Accuracy: {} \n'.format(results[i][0], results[i][1], results[i][2], results[i][3], results[i][4], results[i][5]))
    else:
        print('{}: {}'.format(results[i][0], results[i][5]))



