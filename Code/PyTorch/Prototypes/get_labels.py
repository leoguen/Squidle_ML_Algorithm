import pandas as pd
import os
import glob
import re

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

list_csv = pd.read_csv("/home/ubuntu/IMAS/Code/PyTorch/Annotation_Sets/106569_30_sibling_normalized_list.csv", dtype=str, usecols=['label_name'])
for i in range(len(list_csv)):
    #list_csv.label_name[i] = list_csv.label_name[i].replace('[.]', '_').replace('(', '_').replace(')', '_').replace(' ', '_')
    list_csv.label_name[i] = re.sub("\W", "_", list_csv.label_name[i])
    #list_csv.label_name[i].replace(r'\W', '_')
#print(list_csv.head)
#label_column = list_csv.loc[:,"label_name"]
#classes_df = list_csv.groupby(by=["label_name"]).count()
classes_df = list_csv.pivot_table(index = ['label_name'], aggfunc ='size')

compare_df = pd.concat([img_classes_df, classes_df], axis=1)
#classes_df = label_column.drop_duplicates()
#print(classes_df)
#print(compare_df.shape)
compare_df.to_csv('/home/ubuntu/IMAS/Code/PyTorch/Annotation_Sets/test_compare_sib_classes.csv', header=None, index=True, sep=' ', mode='a')