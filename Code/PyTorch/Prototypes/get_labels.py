import pandas as pd
import os
import glob
import re


#Define List name and which name to use
label_name = "Hard coral cover"
label_name = "Seagrass cover"
label_name = "Macroalgal canopy cover"
label_name = "Ecklonia radiata"

csv_name = 'orig_label_distr_' + label_name
#label.id,label.name,label.uuid,point.id,point.media.deployment.campaign.key,point.media.path_best,point.x,point.y,tag_names
#index,label_name,label_uuid,point_id,point_media_deployment_campaign_key,point_media_path_best,point_x,point_y,tag_names,label_translated_id,label_translated_lineage_names,label_translated_name,label_translated_translation_info,label_translated_uuid

list_csv = pd.read_csv("/pvol/Final_Eck_1_to_10_Database/Original_images/205282_neighbour_Hard_coral_cover_NMSC_list.csv", dtype=str, usecols=['label_name', 'point_media_deployment_campaign_key', 'point_id', 'point_media_path_best'])
list_csv = pd.read_csv("/pvol/Final_Eck_1_to_10_Database/Original_images/22754_neighbour_Seagrass_cover_NMSC_list.csv", dtype=str, usecols=['label_name', 'point_media_deployment_campaign_key', 'point_id', 'point_media_path_best'])
list_csv = pd.read_csv("/pvol/Final_Eck_1_to_10_Database/Original_images/405405_neighbour_Macroalgal_canopy_cover_NMSC_list.csv", dtype=str, usecols=['label_name', 'point_media_deployment_campaign_key', 'point_id', 'point_media_path_best'])
list_csv = pd.read_csv("/pvol/Final_Eck_1_to_10_Database/Original_images/164161_1_to_1_neighbour_Ecklonia_radiata_except.csv", dtype=str, usecols=['label_name', 'point_media_deployment_campaign_key', 'point_id', 'point_media_path_best'])

#Rename the column headers
#list_csv = list_csv.rename(columns={'label.name': 'label_name', 'point.media.deployment.campaign.key': 'point_media_deployment_campaign_key', 'point.id':'point_id', 'point.media.path_best': 'point_media_path_best'})
#Create pivot table
#csv_no_eck = list_csv[list_csv["label_name"].str.contains(label_name) == False] 



#csv_no_eck = list_csv.drop(index='Ecklonia radiata') 
pivot_df = list_csv.pivot_table(index = ['label_name'], aggfunc ='size')
pivot_df = pivot_df.sort_values()
#print(classes_df, len(classes_df))


with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(pivot_df)


print(len(list_csv))
#print(pivot_df[label_name])
#print((pivot_df[label_name]/len(list_csv)))

pivot_df.to_csv(f'/pvol/Annotationset_distr/{csv_name}.csv')
