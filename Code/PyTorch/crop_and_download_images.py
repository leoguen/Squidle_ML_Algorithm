import csv
import os
#from os import path
import pandas as pd
import urllib.request
import cv2
import numpy as np
import re

class crop_download_images():
    
    def create_directory(save_path, dir_name):    
        if os.path.exists(save_path+dir_name) == True:
            print("The directory already exist, no new directory is going to be created.")
        else:
            os.mkdir(save_path+dir_name)
    
    def create_directory_structure(self, save_path):
        if keep_og_size:
            structure_list=[            
            '/Original_images', 
            '/Original_images/' + coi, 
            '/Original_images/Others',
            ]
        else:
            structure_list = [
                '/'+str(bounding_box[0])+'_images', 
                '/'+str(bounding_box[0])+'_images/' + coi, 
                '/'+str(bounding_box[0])+'_images/Others',
                '/'+str(bounding_box[0])+'_images/Padding',
                '/'+str(bounding_box[0])+'_images/Padding/'+coi,
                '/'+str(bounding_box[0])+'_images/Padding/Others']
        for i in structure_list:
            crop_download_images.create_directory(save_path, i)
    
    def download_images(self, save_path, here,bounding_box, list_name):
        # Used to split up data in Ecklonia and Others
        csv_file_df= pd.read_csv(here + list_name)
        csv_file_df.columns = csv_file_df.columns.str.replace('[.]', '_')
        empty_prob_list, crop_prob_list, saving_prob_list = [], [], []
        image_path_before = ''
        for index in range(len(csv_file_df.index)):
            
            image_path, cropped_image, empty_prob_list, crop_prob_list = self.crop_image(save_path,csv_file_df, index, bounding_box, empty_prob_list, crop_prob_list)
            #save image 
            if image_path_before == image_path:
                print('Image dublicate')
                continue
            image_path_before = image_path
            try:
                cv2.imwrite(image_path, cropped_image)
                print("{} | {} Saved: {}".format(bounding_box[0], (index+2), image_path))
            except:
                print('Problem with saving index {}'.format(index))
                saving_prob_list.append([(index+2), csv_file_df.point_media_path_best[index]])
                df = pd.DataFrame(saving_prob_list)
                df.to_csv(save_path + '/' + str(bounding_box[0])+'_images/'+ str(bounding_box[0]) +'_Saving_Problem.csv', index=False) 
        
            empty_prob_list_df = pd.DataFrame(empty_prob_list)
            if keep_og_size:
                empty_prob_list_df.to_csv(save_path + '/Empty_Problem.csv', index=False) 
                crop_prob_list = pd.DataFrame(crop_prob_list)
                crop_prob_list.to_csv(save_path +'/Crop_Problem.csv', index=False) 
            else: 
                empty_prob_list_df.to_csv(save_path + '/' + str(bounding_box[0])+'_images/'+ str(bounding_box[0]) +'_Empty_Problem.csv', index=False) 
                crop_prob_list = pd.DataFrame(crop_prob_list)
                crop_prob_list.to_csv(save_path + '/' + str(bounding_box[0])+'_images/'+ str(bounding_box[0]) +'_Crop_Problem.csv', index=False) 
        print('Problematic files have been saved for {}'.format(bounding_box))

    def get_crop_points(self, x, y, original_image, bounding_box):
        crop_dist = bounding_box[1]/2 
        if x - crop_dist < 0: x0 = 0
        else: x0 = x - crop_dist

        if y - crop_dist < 0: y0 = 0
        else: y0 = y - crop_dist

        if x + crop_dist > original_image.shape[1]: x1 = original_image.shape[1]
        else: x1 = x + crop_dist

        if y + crop_dist > original_image.shape[0]: y1 = original_image.shape[0]
        else: y1 = y + crop_dist

        return  int(x0), int(x1), int(y0), int(y1)

    def crop_image(self, save_path, csv_file_df, index, bounding_box, empty_prob_list, crop_prob_list):
        if csv_file_df.label_name[index] == loi:
            label = coi
        else:
            label = 'Others'
        #create name for the cropped image
        
        cropped_image = []
        try:
            # download the image, convert it to a NumPy array, and then read
            # it into OpenCV format
            resp = urllib.request.urlopen(csv_file_df.point_media_path_best[index])
            array_image = np.asarray(bytearray(resp.read()), dtype=np.uint8)    
            
            # Check if image is not empty
            if array_image.size != 0:
                original_image = cv2.imdecode(array_image, -1) # 'Load it as it is'
                x = original_image.shape[1]*float(csv_file_df.at[index, 'point_x'])
                y = original_image.shape[0]*float(csv_file_df.at[index, 'point_y'])
                
                # get crop coordinates
                x0, x1, y0, y1 = self.get_crop_points(x, y, original_image, bounding_box)
                # crop around coordinates
                #int(y-(bounding_box[0]/2)):int(y+(bounding_box[0]/2)), int(x-(bounding_box[1]/2)):int(x+(bounding_box[1]/2))
                if keep_og_size:cropped_image = original_image
                else:
                    cropped_image = original_image[y0:y1, x0:x1]

                    if list(cropped_image.shape[0:2]) != bounding_box: # Image is not as big as Bounding Box
                        print('Image needs to be padded, Shape {}, ID: {}'.format(cropped_image.shape[0:2],index))
                        label = 'Padding/' + label
                        
            else: # Image is empty
                print('Image is empty ID: {}'.format(index))
                empty_prob_list.append([(index+2), csv_file_df.point_media_path_best[index]])
        except: # Some problem was raised while handling the image
            print('General problem with cropping ID: {}'.format(index))
            crop_prob_list.append([(index+2), csv_file_df.point_media_path_best[index]])
        if keep_og_size:
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
        else: 
            file_name = str(re.sub("\W", "_", csv_file_df.label_name[index])) +"_"+ str(csv_file_df.point_id[index]) +".jpg"

            file_path_and_name = save_path +'/'+str(bounding_box[0])+'_images/' + label + '/' + file_name
        
        return file_path_and_name, cropped_image, empty_prob_list, crop_prob_list

if __name__ == "__main__":
    keep_og_size=True
    coi = 'Seagrass cover' #Ecklonia
    loi = 'Seagrass cover' #Ecklonia radiata
    #Macroalgal canopy cover
    #Hard coral cover
    #Seagrass cover
    download_list = [ 
    #[32,32],
    #[24,24],
    #[64,64],
    #[96,96],
    #[128,128],
    #[160,160],
    #[192,192],
    #[224,224],
    #[256,256], 
    #[288,288],
    #[299,299],
    #[304,304],
    #[336,336],
    #[368,368],
    #[400,400],
    #[432,423],
    #[464,464],
    #[496,496], 
    #[528,528],
    #[560,560], 
    #[592,592],
    #[624,624], 
    [656,656],
    [688,688], 
    [720,720],
    [752,752],
    [784,784], 
    [816,816],
    [848,848], 
    [880,880],
    [912,912],
    [944,944], 
    [976,976], 
    [1008,1008]
    ]
    download_list = [[336,336]]
    path_list = [
        ['NSW_Broughton','/Annotation_Sets/Test_sets/annotations-u45-leo_kelp_AI_test_broughton_is_NSW-leo_kelp_AI_test_broughton_is_25pts-8152-7652a9b48f0e3186fe5d-dataframe.csv'], 
        ['VIC_Discoverybay','/Annotation_Sets/Test_sets/annotations-u45-leo_kelp_AI_test_discoverybay_VIC_phylospora-leo_kelp_AI_test_db_phylospora_25pts-8149-7652a9b48f0e3186fe5d-dataframe.csv'],
        ['TAS_Lanterns','/Annotation_Sets/Test_sets/annotations-u45-leo_kelp_AI_test_lanterns_TAS-leo_kelp_AI_test_lanterns_25pts-8151-7652a9b48f0e3186fe5d-dataframe.csv'], 
        ['VIC_Prom','/Annotation_Sets/Test_sets/annotations-u45-leo_kelp_AI_test_prom_VIC-leo_kelp_AI_test_prom_25pts-8150-7652a9b48f0e3186fe5d-dataframe.csv'], 
        ['WA', '/Annotation_Sets/Test_sets/annotations-u45-leo_kelp_SWC_WA_AI_test-leo_kelp_AI_SWC_WA_test_25pts-8148-7652a9b48f0e3186fe5d-dataframe.csv']]
    
    path_list = [
        ['Seagrass cover','Seagrass_Database' ,'/Annotation_Sets/14961_Seagrass_cover_NMSC_list.csv'], 
        ['Hard coral cover','Hardcoral_Database' ,'/Annotation_Sets/167052_Hard_coral_cover_NMSC_list.csv'],
        ['Macroalgal canopy cover','Macroalgal_Database' ,'/Annotation_Sets/407756_Macroalgal_canopy_cover_NMSC_list.csv']
        ]
    for bounding_box in download_list:
        for name,database,path in path_list:    
            coi = name
            loi = name
            print(bounding_box)
            here = os.path.dirname(os.path.abspath(__file__))
            save_path = '/pvol/' + database
            #save_path = '/pvol/Ecklonia_Database'
            list_name = path

            data = crop_download_images()

            data.create_directory_structure(save_path)
            data.download_images(save_path, here, bounding_box, list_name)
        if keep_og_size:
            break
