import csv
import os
#from os import path
import pandas as pd
import urllib.request
import cv2
import numpy as np

class crop_download_images():
    
    def create_directory(save_path, dir_name):    
        if os.path.exists(save_path+dir_name) == True:
            print("The directory already exist, no new directory is going to be created.")
        else:
            os.mkdir(save_path+dir_name)
    
    def create_directory_structure(self, save_path):
        structure_list = ['/'+str(bounding_box[0])+'_images', '/'+str(bounding_box[0])+'_images/Ecklonia', '/'+str(bounding_box[0])+'_images/Others']
        for i in structure_list:
            crop_download_images.create_directory(save_path, i)
    
    def download_images(self, save_path, here,BOUNDING_BOX, LIST_NAME):
        # Used to split up data in Validation and Training
        csv_file_df= pd.read_csv(here + LIST_NAME)
        csv_file_df.columns = csv_file_df.columns.str.replace('[.]', '_')
        prob_list = []
        for index in range(len(csv_file_df.index)):
            
            image_path, cropped_image, prob_list = self.crop_image(save_path,csv_file_df, index, BOUNDING_BOX, prob_list)
            #save image 
            try:
                cv2.imwrite(image_path, cropped_image)
                print("{} | {} Saved: {}".format(bounding_box[0], index, image_path))
            except:
                print('Problem with saving index {}'.format(index))
                prob_list.append([index, csv_file_df.point_media_path_best[index]])
                df = pd.DataFrame(prob_list)
                df.to_csv('Problematic_files.csv', index=False) 
        
        print('There were problems with the following indeces: {}'.format(prob_list))
        df = pd.DataFrame(prob_list)
        df.to_csv('Problematic_files.csv', index=False) 

    def crop_image(self, save_path, csv_file_df, index, bounding_box, prob_list):
        #dir_name = '/Training/'
        if csv_file_df.label_name[index] == "Ecklonia radiata":
            label = 'Ecklonia'
        else:
            label = 'Others'
        #create name for the cropped image
        file_name = str(csv_file_df.label_name[index].replace('[.]', '_').replace('(', '_').replace(')', '_').replace(' ', '_')) +"_"+ str(csv_file_df.point_id[index]) +".jpg"

        file_path_and_name = save_path +'/'+str(bounding_box[0])+'_images/' + label + '/' + file_name
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
                cropped_image = original_image[int(y-(bounding_box[0]/2)):int(y+(bounding_box[0]/2)), int(x-(bounding_box[1]/2)):int(x+(bounding_box[1]/2))]

                if list(cropped_image.shape[0:2]) != bounding_box: # Image is not as big as Bounding Box
                    print('Image is wrong size, ID: {}'.format(index))
                    prob_list.append([index, csv_file_df.point_media_path_best[index]])
            else: # Image is empty
                print('Problem with downloading id {}'.format(index))
                prob_list.append([index, csv_file_df.point_media_path_best[index]])
        except: # Some problem was raised while handling the image
            print('Problem with cropping id {}'.format(index))
            prob_list.append([index, csv_file_df.point_media_path_best[index]])

        
        return file_path_and_name, cropped_image, prob_list

if __name__ == "__main__":
    #download_list = [[16,16],[32,32], [64,64], [128,128], [256,256], [512,512]]
    download_list = [[24, 24]]
    for bounding_box in download_list:
        #bounding_box = [128,128]
        print(bounding_box)
        here = os.path.dirname(os.path.abspath(__file__))
        save_path = '/pvol'
        list_name = '/Annotation_Sets/41250_NORMALIZED_FULL_ANNOTATION_LIST.csv'

        data = crop_download_images()

        data.create_directory_structure(save_path)
        data.download_images(save_path, here, bounding_box, list_name)