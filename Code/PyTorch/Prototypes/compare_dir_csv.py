import os
import pandas as pd
import re

old_naming_format = False

def get_filenames(root_dir):
    """Get all filenames in all subdirectories of root_dir that end with ".jpg" """
    filenames = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg"):
                if old_naming_format:
                    #if file.startswith('Ecklonia'):
                    # Extract numeric part of filename using regular expressions
                    match = re.search(r'\d+', file)
                    if match:
                        filenames.append(match.group())
                else: 
                    filenames.append(file)
    return filenames

root_dir = "/pvol/Ecklonia_1_to_10_Database/Original_images/All_Images"
df2 = pd.read_csv("/home/ubuntu/Documents/IMAS/Code/PyTorch/Annotation_Sets/588335_1_to_10_Ecklonia_radiata_NMSC_list.csv")
filenames = get_filenames(root_dir)
df2_filenames = []
if old_naming_format:
    for i in range(0, len(filenames)):
        filenames[i] = int(filenames[i])
    df2_point_id = df2['point_id'].tolist()  # Convert 'point_id' column from df2 to a list
else: 
    for index in range(len(df2.point_media_path_best)):
        if '.jpg/' in df2.point_media_path_best[index]:
            second_part = str(re.sub(".*/(.*).jpg/", "\\1", df2.point_media_path_best[index])) 
        elif '.jpg' in df2.point_media_path_best[index]:
            second_part = str(re.sub(".*/(.*).jpg", "\\1", df2.point_media_path_best[index]))
        elif '.JPG/' in df2.point_media_path_best[index]:
            second_part = str(re.sub(".*/(.*).JPG/", "\\1", df2.point_media_path_best[index])) 
        elif '.JPG' in df2.point_media_path_best[index]:
            second_part = str(re.sub(".*/(.*).JPG", "\\1", df2.point_media_path_best[index])) 
        else:
            second_part =str(re.sub(".*/(.*)", "\\1", df2.point_media_path_best[index]))

        file_name = str(re.sub("\W", "_", df2.point_media_deployment_campaign_key[index])) +"-"+ re.sub("\W", "_",second_part) + '.jpg'
        df2_filenames.append(file_name)
        print(len(df2_filenames), end="\r", flush=True )
if old_naming_format:
    num_in_filenames = len(set(df2_point_id).intersection(filenames))
    num_not_in_filenames = len(set(df2_point_id).difference(filenames))
else:
    num_in_filenames = len(set(df2_filenames).intersection(filenames))
    num_not_in_filenames = len(set(df2_filenames).difference(filenames))

print(f"Number of entries in df2_point_id that are in filenames: {num_in_filenames}")
print(f"Number of entries in df2_point_id that are not in filenames: {num_not_in_filenames}")
