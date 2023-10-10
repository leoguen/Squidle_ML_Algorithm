import csv
import os
#from os import path
import pandas as pd
import urllib.request
import cv2
import numpy as np
import re

class crop_download_images():

    def get_downloaded_files(self, dir_path, csv_path):
        # Load the CSV file into a pandas dataframe
        df = pd.read_csv(csv_path)

        dir_path = dir_path + '/Original_images/All_Images/'
        # Create a pandas dataframe from the .jpg files in save_path
        jpg_files_df = pd.DataFrame({'filename': [f for f in os.listdir(dir_path) if f.endswith('.jpg')]})

        # Drop duplicates in the jpg_files_df based on the media_path column
        df.drop_duplicates(subset=['point_media_path_best'], inplace=True)
        df['filename'] = None
        df = df.reset_index(drop=True)
        print('Looking up differences between images in csv and images in directory.')
        for index in range(len(df)):
            print(f'{index}/{len(df)}', end='\r')

            if '.jpg/' in df.point_media_path_best[index]:
                second_part = str(re.sub(".*/(.*).jpg/", "\\1", df.point_media_path_best[index])) 
            elif '.jpg' in df.point_media_path_best[index]:
                second_part = str(re.sub(".*/(.*).jpg", "\\1", df.point_media_path_best[index]))
            elif '.JPG/' in df.point_media_path_best[index]:
                second_part = str(re.sub(".*/(.*).JPG/", "\\1", df.point_media_path_best[index])) 
            elif '.JPG' in df.point_media_path_best[index]:
                second_part = str(re.sub(".*/(.*).JPG", "\\1", df.point_media_path_best[index])) 
            else:
                second_part =str(re.sub(".*/(.*)", "\\1", df.point_media_path_best[index]))
            df.filename[index] = str(re.sub("\W", "_", df.point_media_deployment_campaign_key[index])) +"-"+ re.sub("\W", "_",second_part) + '.jpg'
        # Combine the campaign key and second part to create the filename
        #df['filename'] = df['point_media_deployment_campaign_key'].str.replace(r'\W', '_') + '-' + df['second_part'].str.replace(r'\W', '_') + '.jpg'

        # Find the set difference between df['filename'] and jpg_files_df['filename']
        print(f'The original file has {len(df)} entries.')
        df = df[~df['filename'].isin(set(jpg_files_df['filename']))]
        print(f'The residual file has {len(df)} entries.')
        
        df.drop_duplicates(subset=['point_media_path_best'], inplace=True)
        df = df.reset_index(drop=True)
        return df

    def create_directory(save_path, dir_name):    
        if os.path.exists(save_path+dir_name) == True:
            print("The directory already exist, no new directory is going to be created.")
        else:
            os.mkdir(save_path+dir_name)
    
    def create_directory_structure(self, save_path):

        structure_list=[            
        '/Original_images', 
        '/Original_images/All_Images', 
        ]
        for i in structure_list:
            crop_download_images.create_directory(save_path, i)
    
    def download_images(self, save_path, list_name):
        # Used to split up data in Ecklonia and Others
        if isinstance(list_name, str):
            csv_file_df = pd.read_csv(list_name)
        else:
            csv_file_df = list_name
        csv_file_df.columns = csv_file_df.columns.str.replace('[.]', '_')
        csv_file_df = csv_file_df.sort_values(by=['point_media_path_best'])
        empty_prob_list, crop_prob_list, saving_prob_list = [], [], []
        image_path_before = ''
        for index in range(len(csv_file_df.index)):
            image_path, cropped_image, empty_prob_list, crop_prob_list, error = self.check_image(save_path,csv_file_df, index, empty_prob_list, crop_prob_list)
            #save image 
            if image_path_before == image_path:
                print('Image dublicate')
                continue
            image_path_before = image_path
            # Only try to save the image if there was no error in check_image
            if error == False:
                try:
                    cv2.imwrite(image_path, cropped_image)
                    print(f"{[0]} | {(index+2)}/{len(csv_file_df.index)} Saved: {image_path}")
                except:
                    print('Problem with saving index {}'.format(index))
                    saving_prob_list.append([(index+2), csv_file_df.point_media_path_best[index]])
                    df = pd.DataFrame(saving_prob_list)
                    df.to_csv(save_path + '/Saving_Problem.csv', index=False) 
            
            empty_prob_list_df = pd.DataFrame(empty_prob_list)
            empty_prob_list_df.to_csv(save_path + '/Empty_Problem.csv', index=False) 
            crop_prob_list = pd.DataFrame(crop_prob_list)
            crop_prob_list.to_csv(save_path +'/Crop_Problem.csv', index=False) 
        print('Problematic files have been saved')

    def check_image(self, save_path, csv_file_df, index, empty_prob_list, crop_prob_list):
        label = 'All_Images'
        error = False
        try:
            # download the image, convert it to a NumPy array, and then read
            # it into OpenCV format
            resp = urllib.request.urlopen(csv_file_df.point_media_path_best[index])
            array_image = np.asarray(bytearray(resp.read()), dtype=np.uint8)    
            # Check if image is not empty
            if array_image.size != 0:
                original_image = cv2.imdecode(array_image, -1) # 'Load it as it is'
                #print(csv_file_df.point_media_path_best[index])
                if '.jpg/' in csv_file_df.point_media_path_best[index]:
                    second_part = str(re.sub(".*/(.*).jpg/", "\\1", csv_file_df.point_media_path_best[index])) 
                elif '.jpg' in csv_file_df.point_media_path_best[index]:
                    second_part = str(re.sub(".*/(.*).jpg", "\\1", csv_file_df.point_media_path_best[index]))
                elif '.JPG/' in csv_file_df.point_media_path_best[index]:
                    second_part = str(re.sub(".*/(.*).JPG/", "\\1", csv_file_df.point_media_path_best[index])) 
                elif '.JPG' in csv_file_df.point_media_path_best[index]:
                    second_part = str(re.sub(".*/(.*).JPG", "\\1", csv_file_df.point_media_path_best[index])) 
                else:
                    second_part =str(re.sub(".*/(.*)", "\\1", csv_file_df.point_media_path_best[index]))

                file_name = str(re.sub("\W", "_", csv_file_df.point_media_deployment_campaign_key[index])) +"-"+ re.sub("\W", "_",second_part)
                file_path_and_name = save_path +'/Original_images/' + label + '/' + file_name +'.jpg'
                
                return file_path_and_name, original_image, empty_prob_list, crop_prob_list, error
            else: # Image is empty
                print('Image is empty ID: {}'.format(index))
                empty_prob_list.append([(index+2), csv_file_df.point_media_path_best[index]])
                error = True
                return file_path_and_name, original_image, empty_prob_list, crop_prob_list, error
        except: # Some problem was raised while handling the image
            print('General problem with cropping ID: {}'.format(index))
            crop_prob_list.append([(index+2), csv_file_df.point_media_path_best[index]])
            error = True
            return file_path_and_name, original_image, empty_prob_list, crop_prob_list, error
        

        

if __name__ == "__main__":
    coi = 'Hard coral cover' #'Seagrass cover' 
    loi = 'Hard coral cover' #Seagrass cover' 
    #Macroalgal canopy cover
    #Hard coral cover
    #Seagrass cover
    
    path_list = [
        ['Seagrass cover','Final_Eck_1_to_10_Database' ,'/Annotation_Sets/Final_Sets/22754_neighbour_Seagrass_cover_NMSC_list.csv'], 
        ['Hard coral cover','Final_Eck_1_to_10_Database' ,'/Annotation_Sets/Final_Sets/205282_neighbour_Hard_coral_cover_NMSC_list.csv'],
        ['Macroalgal canopy cover','Final_Eck_1_to_10_Database' ,'/Annotation_Sets/Final_Sets/405405_neighbour_Macroalgal_canopy_cover_NMSC_list.csv']
        ]
    
    """
    path_list = [
        #['Seagrass cover', 'NMSC_Testbase/Seagrass_Port_Phillip_2017', '/Annotation_Sets/Final_Sets/Seagrass_RLS_Port Phillip Bay_2017.csv'], 
        #['Seagrass cover', 'NMSC_Testbase/Seagrass_Port_Phillip_2016', '/Annotation_Sets/Final_Sets/Seagrass_RLS_Port Phillip Bay_2016.csv'],
        ['Seagrass cover', 'NMSC_Testbase/Seagrass_Port_Phillip_2010', '/Annotation_Sets/Final_Sets/Seagrass_RLS_Port Phillip Heads_2010.csv'],
        ['Macroalgal cover', 'NMSC_Testbase/Macroalgal_Gulf', '/Annotation_Sets/Final_Sets/Macroalgal_RLS_Gulf St Vincent_2012.csv'],
        ['Macroalgal cover', 'NMSC_Testbase/Marcoalgal_Jervis', '/Annotation_Sets/Final_Sets/Macroalgal_RLS_Jervis Bay Marine Park_2015.csv'],
        ['Macroalgal cover', 'NMSC_Testbase/Macroalgal_Port_Phillip', '/Annotation_Sets/Final_Sets/Macroalgal_RLS_Port Phillip Bay_2017.csv'],
        ['Hardcoral cover', 'NMSC_Testbase/Hardcoral_Queensland', '/Annotation_Sets/Final_Sets/Hardcoral_RLS_Queensland (other)_2015.csv'],
        ['Hardcoral cover', 'NMSC_Testbase/Hardcoral_Norfolk', '/Annotation_Sets/Final_Sets/Hardcoral_RLS_Norfolk Island_2013.csv'],
        ['Hardcoral cover', 'NMSC_Testbase/Hardcoral_Ningaloo', '/Annotation_Sets/Final_Sets/Hardcoral_RLS_Ningaloo Marine Park_2016.csv']]
    
    path_list = [
        ['Ecklonia radiata', 'Final_Eck_1_to_10_Database', '/Annotation_Sets/Final_Sets/405405_neighbour_Macroalgal_canopy_cover_NMSC_list.csv'],
        ['Ecklonia radiata', 'Final_Eck_1_to_10_Database', '/Annotation_Sets/Final_Sets/205282_neighbour_Hard_coral_cover_NMSC_list.csv'],
        ['Ecklonia radiata', 'Final_Eck_1_to_10_Database', '/Annotation_Sets/Final_Sets/22754_neighbour_Seagrass_cover_NMSC_list.csv']]
    """
    path_list = [['/pvol/Final_Eck_1_to_10_Database', '/pvol/Final_Eck_1_to_10_Database/Original_images/1231165_neighbour_Sand _ mud (_2mm)_list.csv']]

    for save_path, path in path_list:
        #save_path = '/pvol/Ecklonia_Database'
        list_name = path

        data = crop_download_images()
        data.create_directory_structure(save_path)
        residual_csv = data.get_downloaded_files(save_path, path)
        data.download_images(save_path, residual_csv)
