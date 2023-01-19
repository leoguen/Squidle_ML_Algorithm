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
save_path = '/pvol/random_validation/'+str(bounding_box[0])+'_images'

# Get list of models in same file tree
model_list = [f for f in listdir(HERE + '/models') if isfile(join(HERE + '/models', f))]
# Delete all models with wrong ending
model_list = [val for val in model_list if val.endswith(".pth")]
model_list = [HERE + '/models/' +s for s in model_list]


# Get models from /pvol/
for root, dirs, files in os.walk('/pvol/models/', topdown=False):
    for name in files:
        if name.endswith('.pt'):
            model_list.append(root + name)

#print('This is the modellist: {}'.format(model_list))

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
    check = 0
    if model_name.endswith('.pth'):
        # For Model trained on CPU and executed on CPU
        loaded_model = torch.jit.load(model_name, map_location='cpu')
        loaded_model.eval()
        check = 1
        
    elif model_name.endswith('.pt'):
        check = 1
        print(model_name)
        loaded_model = torch.load(model_name)
        loaded_model.eval()

        print('Functionality needs to be added.')
    else:
        check = 0
        loaded_model = 0
        print('Weird fileformat.')

    
    return check, loaded_model


# Iterate through list of filenames and make predictions for each
for index, model_name in enumerate(model_list):
    check, loaded_model = load_model(model_name)
    if check == 0:
        print('model {} could not be loaded.'.format(model_name))
        continue
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
                print('Index {} out of {}.'.format(img_index, len(eck_files)+len(other_files)),  end="\r", flush=True)
            except: # image needs to be reshaped for CNN
                print('CNN index {} out of {}.'.format(img_index, len(eck_files)+len(other_files)),  end="\r", flush=True)
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
    print('{} accuracy: {}'.format(model_name, accuracy))
    results[index] = model_name, truePos, trueNeg, falsePos, falseNeg, accuracy
    accuracy, truePos, trueNeg, falsePos, falseNeg = 0, 0, 0, 0, 0

results = sorted(results, key=lambda x: x[5], reverse=True)
for i in range(len(model_list)):
    if i == 0:
        print(' Best model {}: \n No. of True Positives: {} \n No. of True Negatives: {} \n No. of False Positives: {} \n No. of False Negatives: {} \n Accuracy: {} \n'.format(results[i][0], results[i][1], results[i][2], results[i][3], results[i][4], results[i][5]))
    else:
        print('{}: {}'.format(results[i][0], results[i][5]))



