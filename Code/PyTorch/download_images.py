import os
import pandas as pd
import urllib.request
import cv2
import numpy as np
import re
import argparse

class crop_download_images():

    def __init__(self, args):
        self.save_path = args.save_path
        self.csv_path = args.csv_path
    
    def get_downloaded_files(self, dir_path, csv_path):
        # Load the CSV file into a pandas dataframe
        df = pd.read_csv(csv_path, low_memory=False)

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
            df.loc[index, 'filename'] = str(re.sub("\W", "_", df.loc[index, 'point_media_deployment_campaign_key'])) + "-" + re.sub("\W", "_", second_part) + '.jpg'
        # Combine the campaign key and second part to create the filename
        #df['filename'] = df['point_media_deployment_campaign_key'].str.replace(r'\W', '_') + '-' + df['second_part'].str.replace(r'\W', '_') + '.jpg'

        # Find the set difference between df['filename'] and jpg_files_df['filename']
        print(f'The original file has {len(df)} entries.')
        df = df[~df['filename'].isin(set(jpg_files_df['filename']))]
        print(f'The residual file has {len(df)} entries.')
        
        df.drop_duplicates(subset=['point_media_path_best'], inplace=True)
        df = df.reset_index(drop=True)
        return df
    """
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
    """
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
                file_path_and_name = save_path + file_name +'.jpg'
                
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
        

        
def get_args():
    parser = argparse.ArgumentParser(description="Crop and Download Images")
    parser.add_argument('--save_path', default='/home/leo/Documents/IMAS/Git_Test/IMAS/Code/PyTorch/Images/', help="Save path (default: './Images/')")
    parser.add_argument('--csv_path', default='/home/leo/Documents/IMAS/Git_Test/IMAS/Code/PyTorch/Annotationsets/29219_neighbour_Sand _ mud (_2mm).csv', help="CSV file path")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    data = crop_download_images(args)

    #data.create_directory_structure(data.save_path)
    residual_csv = data.get_downloaded_files(data.save_path, data.csv_path)
    data.download_images(data.save_path, residual_csv)
