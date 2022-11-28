import csv
import os
from os import path
import pandas as pd
import urllib.request
import cv2
import numpy as np

class crop_download_images():
    
    def create_directory(HERE, dir_name):    
        if os.path.exists(HERE+dir_name) == True:
            print("The directory already exist, no new directory is going to be created.")
        else:
            os.mkdir(HERE+dir_name)
    
    def create_directory_structure(self, HERE):
        structure_list = ['/images', '/images/Validation', '/images/Validation/Ecklonia', '/images/Validation/Others', '/images/Training', '/images/Training/Ecklonia', '/images/Training/Others']
        for i in structure_list:
            crop_download_images.create_directory(HERE, i)
    
    def download_images(self, HERE, BOUNDING_BOX, LIST_NAME):
        # Used to split up data in Validation and Training
        COUNTER = [0,0]
        csv_file_df= pd.read_csv(HERE + LIST_NAME)
        csv_file_df.columns = csv_file_df.columns.str.replace('[.]', '_')

        for index in range(len(csv_file_df.index)):
            
            image_path, cropped_image, COUNTER = self.crop_image(HERE,csv_file_df, index, BOUNDING_BOX, COUNTER)
            #save image 
            cv2.imwrite(image_path, cropped_image)
            print("Saved: " + image_path)
            print(COUNTER)

    def crop_image(self, HERE, csv_file_df, index, BOUNDING_BOX, COUNTER):
        dir_name = '/Training/'
        if csv_file_df.label_uuid[index] == "5b3ca3b9-20c0-4a7f-a08e-14ebd493f9cf":
            label = 'Ecklonia'
            COUNTER[0] += 1
            if COUNTER[0] % 5 == 0:
                dir_name = '/Validation/'
        else:
            label = 'Others'
            COUNTER[1] += 1
            if COUNTER[1] % 5 == 0:
                dir_name = '/Validation/'
        #create name for the cropped image
        file_name = str(csv_file_df.point_media_id[index]) +"_"+ str(csv_file_df.point_id[index]) +".jpg"

        file_path_and_name = HERE +'/images' + dir_name + label + '/' + file_name
        
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        resp = urllib.request.urlopen(csv_file_df.point_media_path_best[index])
        array_image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        original_image = cv2.imdecode(array_image, -1) # 'Load it as it is'
        
        # get dimension of image and multiply with point coordinates
        x = original_image.shape[1]*csv_file_df.point_x[index]
        y = original_image.shape[0]*csv_file_df.point_y[index]
        # crop around coordinates
        cropped_image = original_image[int(y-(BOUNDING_BOX[0]/2)):int(y+(BOUNDING_BOX[0]/2)), int(x-(BOUNDING_BOX[1]/2)):int(x+(BOUNDING_BOX[1]/2))]
        
        return file_path_and_name, cropped_image, COUNTER

if __name__ == "__main__":
    BOUNDING_BOX = [24,24]
    HERE = os.path.dirname(os.path.abspath(__file__))
    LIST_NAME = '/Annotation_Sets/annotations-u45-Lanterns_shallow_2012_kelp_only_annotations_21-30m-Tomas-4058-53007f4f89e836d1c9bc-dataframe.csv'

    data = crop_download_images()

    data.create_directory_structure(HERE)
    data.download_images(HERE, BOUNDING_BOX, LIST_NAME)