import csv
import os
from os import path
import pandas as pd
import urllib.request
import cv2
import numpy as np

class extract_metadata():
    def procedure(self):
        #Global Variables
        BOUNDING_BOX_SIZE = [24,24]

        # folder path
        dir_path = r'/home/leo/Documents/IMAS/Code/Extract_Metadata/Metadata'

        # list to store metadata filenames (for multiple metadata files)
        csv_list = []

        # Iterate over directory to find all metadata files and save them
        for path in os.listdir(dir_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)):
                csv_list.append(path)

        #read first metadata file
        metadata= pd.read_csv(os.path.join(path, dir_path, csv_list[0]))

        # "." needs to be replaced by "_" otherwise panda throws an error 
        metadata.columns = metadata.columns.str.replace('[.]', '_')

        #create sub-dataset with only Ecklonia radiata entries
        #ecklonia_df = metadata[metadata.label_name == "Ecklonia radiata" ]

        #create a directory which contains only images that have an Ecklonia radiata annotation 
        dir_name = "images_" + csv_list[0] 

        #check if directory already exist
        if os.path.exists(str(os.path.join(dir_path, dir_name))) == True:
            print("The directory already exist, no new directory is going to be created.")
        else:
            os.mkdir(os.path.join(dir_path, dir_name))

        #save dataframe with only Ecklonia entries in it
        #ecklonia_df.to_csv(os.path.join(dir_path, dir_name)+'/Ecklonia_' + csv_list[0] +'.csv', index=False)

        #create .csv file for PyTorch
        f = open(os.path.join(dir_path, "Pytorch_Ecklonia.csv"), 'w')

        for index in range(len(metadata.index)):
            path_and_cropped_image = self.crop_image(metadata, index, 
                                    dir_path, dir_name, BOUNDING_BOX_SIZE)
            #save image 
            cv2.imwrite(path_and_cropped_image[0], path_and_cropped_image[1])
            print("Saved: " + path_and_cropped_image[0])

            #add to .csvfile
            self.save_csv(dir_path, path_and_cropped_image[2], path_and_cropped_image[3])

    def save_csv(self, dir_path, label, file_name):
        label_id = 0 # means others
        if label == "Ecklonia": label_id = 1
        #open and add to .csv dataset for Pytorch
        f = open(os.path.join(dir_path, "Pytorch_Ecklonia.csv"), 'a')

        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerow([file_name , label_id])

        # close the file
        f.close()

    def crop_image(self, dataframe, index, dir_path, dir_name, BOUNDING_BOX_SIZE):
        if dataframe.label_name[index] == "Ecklonia radiata":
            label = "Ecklonia"
            #ecklonia_index +=1
        else:
            label = "Others"
            #others_index +=1
        #create name for the cropped image
        file_name = label +"_"+ dataframe.point_media_deployment_campaign_key[index] +"_"+ str(index) +".jpg"
        
        file_path_and_name = os.path.join(dir_path, dir_name, file_name )
        
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        resp = urllib.request.urlopen(dataframe.point_media_path_best[index])
        array_image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        original_image = cv2.imdecode(array_image, -1) # 'Load it as it is'
        
        # get dimension of image and multiply with point coordinates
        x = original_image.shape[1]*dataframe.point_x[index]
        y = original_image.shape[0]*dataframe.point_y[index]
        # crop around coordinates
        cropped_image = original_image[int(y-(BOUNDING_BOX_SIZE[0]/2)):int(y+(BOUNDING_BOX_SIZE[0]/2)), int(x-(BOUNDING_BOX_SIZE[1]/2)):int(x+(BOUNDING_BOX_SIZE[1]/2))]
        return file_path_and_name, cropped_image, label, file_name

if __name__ == "__main__":
    data = extract_metadata()
    data.procedure()