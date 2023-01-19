import os
import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO
from skimage import io


class kelp_dataset_generator_https(Dataset):

    
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        BOUNDING_BOX_SIZE = [24,24]
        image_path = self.annotations.iloc[index,11]
        response = requests.get(image_path)
        image = np.asarray(Image.open(BytesIO(response.content)))

        # get dimension of image and multiply with point coordinates
        x = image.shape[1]*self.annotations.iloc[index, 17]
        y = image.shape[0]*self.annotations.iloc[index, 18]
        
        # crop around coordinates
        cropped_image = image[int(y-(BOUNDING_BOX_SIZE[0]/2)):int(y+(BOUNDING_BOX_SIZE[0]/2)), int(x-(BOUNDING_BOX_SIZE[1]/2)):int(x+(BOUNDING_BOX_SIZE[1]/2))]
        cropped_image = Image.fromarray(np.uint8(cropped_image))
        #image_path = os.path.join(self.root_dir, self.annotations.iloc[index,0])
        #image = io.imread(image_path)
        group = 0
        if self.annotations.iloc[index, 4] == 'Ecklonia radiata':
            group = 1
        #print('The group is: ' + str(group) + ' The size is: ' + str(cropped_image.size[0])+ 'by ' + str(cropped_image.size[1]))
        y_label = torch.tensor(group)
        print(str(index) + ' : ' + image_path)

        if self.transform:
            cropped_image = self.transform(cropped_image)
        
        return (cropped_image, y_label)