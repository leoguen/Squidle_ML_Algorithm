import pandas as pd
import os
import glob
import re
'''
path = '/pvol/Ecklonia_Sibling_Database/299_images/**/*.jpg'
#extension = 'jpg'
#os.chdir(path)
img_names = glob.glob(path, recursive=True)
for i in range(len(img_names)):
    img_names[i] = re.sub(".*/(.*)", "\\1", img_names[i])
    img_names[i] = re.sub("(.*)_\d+.jpg", "\\1", img_names[i])
#print(configfiles)
img_names_df = pd.DataFrame (img_names,columns = ['img_name'] )
img_classes_df = img_names_df.pivot_table(index = ['img_name'], aggfunc ='size')
print(img_classes_df)
img_classes_df.to_csv('/home/ubuntu/IMAS/Code/PyTorch/Annotation_Sets/0_compare_sib_classes.csv', header=None, index=True, sep=' ', mode='a')
'''
#Define List name and which name to use
csv_name = '1_to_1_Ecklonia_radiata'
label_name = "Macroalgal canopy cover"
label_name = "Seagrass cover"
label_name = "Hard coral cover"
label_name = "Ecklonia radiata"
#label.id,label.name,label.uuid,point.id,point.media.deployment.campaign.key,point.media.path_best,point.x,point.y,tag_names
#index,label_name,label_uuid,point_id,point_media_deployment_campaign_key,point_media_path_best,point_x,point_y,tag_names,label_translated_id,label_translated_lineage_names,label_translated_name,label_translated_translation_info,label_translated_uuid
list_csv = pd.read_csv("/pvol/Final_Eck_1_to_10_Database/Original_images/680037_1_to_10_Ecklonia_radiata_except.csv", dtype=str, usecols=['label_name', 'point_media_deployment_campaign_key', 'point_id', 'point_media_path_best'])

#Rename the column headers
#list_csv = list_csv.rename(columns={'label.name': 'label_name', 'point.media.deployment.campaign.key': 'point_media_deployment_campaign_key', 'point.id':'point_id', 'point.media.path_best': 'point_media_path_best'})
#Create pivot table
csv_no_eck = list_csv[list_csv["label_name"].str.contains(label_name) == False] 



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
print(pivot_df[label_name])
print((pivot_df[label_name]/len(list_csv)))

#pivot_df.to_csv(f'/home/ubuntu/Documents/IMAS/Code/PyTorch/Annotation_Sets/Paper_Data/{csv_name}.csv')
