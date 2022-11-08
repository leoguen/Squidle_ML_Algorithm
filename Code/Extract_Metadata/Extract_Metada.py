import csv
import os
from os import path
import pandas as pd
import urllib.request
import cv2
import numpy as np

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
ecklonia_df = metadata[metadata.label_name == "Ecklonia radiata" ]

#create a directory which contains only images that have an Ecklonia radiata annotation 
dir_name = "images_" + csv_list[0] 

#check if directory already exist
if os.path.exists(str(os.path.join(dir_path, dir_name))) == True:
    print("The directory already exist, no new directory is going to be created.")
else:
    os.mkdir(os.path.join(dir_path, dir_name))

#save dataframe with only Ecklonia entries in it
ecklonia_df.to_csv(os.path.join(dir_path, dir_name)+'/Ecklonia_' + csv_list[0] +'.csv', index=False)

for index in ecklonia_df.index:
    #create name for the cropped image
    file_name = ecklonia_df.point_media_deployment_campaign_key[index] +"_"+ str(index) +".jpg"
    
    file_path_and_name = os.path.join(dir_path, dir_name, file_name )
    
    # download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
    resp = urllib.request.urlopen(ecklonia_df.point_media_path_best[index])
    array_image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    original_image = cv2.imdecode(array_image, -1) # 'Load it as it is'
    
    # get dimension of image and multiply with point coordinates
    x = original_image.shape[1]*ecklonia_df.point_x[index]
    y = original_image.shape[0]*ecklonia_df.point_y[index]
    # crop around coordinates
    cropped_image = original_image[int(y-(BOUNDING_BOX_SIZE[0]/2)):int(y+(BOUNDING_BOX_SIZE[0]/2)), int(x-(BOUNDING_BOX_SIZE[1]/2)):int(x+(BOUNDING_BOX_SIZE[1]/2))]
    
    #save image 
    cv2.imwrite(file_path_and_name, cropped_image)
    print("Saved: " + file_path_and_name)
