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

#create another subset only with url and uuid label
#url_df = ecklonia_df.loc[ecklonia_df[],"point_media_path_best" "label_uuid"]

#download image from webiste and name it according to label.uuid


for index in ecklonia_df.index:
    
    ###BEGIN TODO###
    #!!!Problem is label uuid is not refering to image but to label -> all have the same label, maybe work with image name?
    #Try to save multiple image from dataset and center around point saved in metadata set
    #Look at Bot Code
    ###END TODO###
    file_name = ecklonia_df.label_uuid[index] +".jpg"
    print(index)
    print(file_name)
    print(ecklonia_df.label_uuid[19])

    file_path_and_name = os.path.join(dir_path, dir_name, file_name )
    # download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
    resp = urllib.request.urlopen(ecklonia_df.point_media_path_best[index])
    array_image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    original_image = cv2.imdecode(array_image, -1) # 'Load it as it is'
    
    
    #urllib.request.urlretrieve(ecklonia_df.point_media_path_best[index], file_path_and_name)
    #original_image= cv2.imread(file_path_and_name)
    
    center = original_image.shape
    x = center[1]/2
    y = center[0]/2
    cropped_image = original_image[int(y-(BOUNDING_BOX_SIZE[0]/2)):int(y+(BOUNDING_BOX_SIZE[0]/2)), int(x-(BOUNDING_BOX_SIZE[1]/2)):int(x+(BOUNDING_BOX_SIZE[1]/2))]
    
    cv2.imwrite(file_path_and_name, cropped_image)
    print("Saved: " + file_path_and_name)
    #break