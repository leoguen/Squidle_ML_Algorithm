import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from skimage import io
import matplotlib as plt
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = "Optuna/optuna_simple_prototype/best_trial_Nov-18-2022-15-54_88.pickle"

# Loading model to compare the results
model = pickle.load(open(PATH,'rb'))


#model = torch.load("/home/leo/Documents/IMAS/Code/Optuna/optuna_simple_prototype/0_trial.pickle")
#model.eval()
predictions=model.predict('/home/leo/Documents/IMAS/Code/Optuna/optuna_simple_prototype/Ecklonia_dataset/Ecklonia_Tasmania201006_0.jpg')
print(predictions)
for param in model.parameters():
    print(param)







"""
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    tensor=tensor.to(device)
    output = model.forward(tensor)
    
    probs = torch.nn.functional.softmax(output, dim=1)
    conf, classes = torch.max(probs, 1)
    return conf.item(), index_to_breed[classes.item()]

image_path="/content/test/06b3a4da7b96404349e51551bf611551.jpg"
image = plt.imread(image_path)
plt.imshow(image)

with open(image_path, 'rb') as f:
    image_bytes = f.read()

    conf,y_pre=get_prediction(image_bytes=image_bytes)
    print(y_pre, ' at confidence score:{0:.2f}'.format(conf))
"""

"""
pickled_model = pickle.load(open('best_trial_Nov-10-2022-11-29_88.pickle', 'rb'))
classifier_code = "ECK" 
# load the model from disk
image = io.imread("Ecklonia_dataset/Ecklonia_Tasmania201006_0.jpg")
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)

prob = pickled_model.predict(image)
print(prob)
"""