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
        prob_list = []
        for index in range(len(csv_file_df.index)):
            
            image_path, cropped_image, COUNTER, prob_list = self.crop_image(HERE,csv_file_df, index, BOUNDING_BOX, COUNTER, prob_list)
            #save image 
            try:
                cv2.imwrite(image_path, cropped_image)
                print("Saved: " + image_path)
                print(COUNTER)
            except:
                print('Problem with saving index {}'.format(index))
                prob_list.append([index, csv_file_df.point_media_path_best[index]])
                df = pd.DataFrame(prob_list)
                df.to_csv('Problematic_files.csv', index=False) 
        
        print('There were problems with the following indeces: {}'.format(prob_list))
        df = pd.DataFrame(prob_list)
        df.to_csv('Problematic_files.csv', index=False) 

    def crop_image(self, HERE, csv_file_df, index, BOUNDING_BOX, COUNTER, prob_list):
        dir_name = '/Training/'
        if csv_file_df.label_name[index] == "Ecklonia radiata":
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
        file_name = str(csv_file_df.label_name[index].replace('[.]', '_').replace('(', '_').replace(')', '_').replace(' ', '_')) +"_"+ str(csv_file_df.point_id[index]) +".jpg"

        file_path_and_name = HERE +'/images' + dir_name + label + '/' + file_name
        try:
            # download the image, convert it to a NumPy array, and then read
            # it into OpenCV format
            resp = urllib.request.urlopen(csv_file_df.point_media_path_best[index])
            array_image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            cropped_image = []
            
            # Check if image is not empty
            if array_image.size != 0:
                #if array_image.shape != BOUNDING_BOX:
                #    print(array_image.size)
                #    print(array_image.shape)
                #    array_image = np.pad(array_image, BOUNDING_BOX - array_image.shape, 'constant', constant_values=(0))
                original_image = cv2.imdecode(array_image, -1) # 'Load it as it is'
                x = original_image.shape[1]*float(csv_file_df.at[index, 'point_x'])
                y = original_image.shape[0]*float(csv_file_df.at[index, 'point_y'])
                # crop around coordinates
                cropped_image = original_image[int(y-(BOUNDING_BOX[0]/2)):int(y+(BOUNDING_BOX[0]/2)), int(x-(BOUNDING_BOX[1]/2)):int(x+(BOUNDING_BOX[1]/2))]

                if list(cropped_image.shape[0:2]) != BOUNDING_BOX: # Image is not as big as Bounding Box
                    # Reset Counter so that distribution does not get mixed up
                    cropped_image = [] 
                    if label == 'Ecklonia': COUNTER[0] -= 1
                    if label == 'Others': COUNTER[1] -= 1 
                    print('Image is wrong size, ID: {}'.format(index))
                    prob_list.append([index, csv_file_df.point_media_path_best[index]])
            else: # Image is empty
                # Reset Counter so that distribution does not get mixed up
                if label == 'Ecklonia': COUNTER[0] -= 1
                if label == 'Others': COUNTER[1] -= 1 
                print('Problem with downloading id {}'.format(index))
                prob_list.append([index, csv_file_df.point_media_path_best[index]])
        except: # Some problem was raised while handling the image
            # Reset Counter so that distribution does not get mixed up
            if label == 'Ecklonia': COUNTER[0] -= 1
            if label == 'Others': COUNTER[1] -= 1 
            print('Problem with cropping id {}'.format(index))
            prob_list.append([index, csv_file_df.point_media_path_best[index]])

        
        return file_path_and_name, cropped_image, COUNTER, prob_list

if __name__ == "__main__":
    BOUNDING_BOX = [224,224]
    HERE = os.path.dirname(os.path.abspath(__file__))
    LIST_NAME = '/Annotation_Sets/41250_NORMALIZED_FULL_ANNOTATION_LIST.csv'

    data = crop_download_images()

    data.create_directory_structure(HERE)
    data.download_images(HERE, BOUNDING_BOX, LIST_NAME)