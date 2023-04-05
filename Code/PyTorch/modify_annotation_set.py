import csv
import os
from os import path
import pandas as pd
import urllib.request
import cv2
import numpy as np

class modify_annotation_set():

    def __init__(self):
        self.sibling = False
        self.red_list = True
        self.sib_factor = 0.3
        self.coi = 'Seagrass cover' #Class of Interest
        #Seagrass cover 
        #Macroalgal canopy cover
        #Hard coral cover
        
        self.row_name = 'label_translated_name' #'label_name' 
        self.norm_factor = 1

    def delete_entries(self, csv_file_df, label, value):
        shape_before = csv_file_df.shape[0]
        csv_file_df = csv_file_df.loc[csv_file_df[label] != value]
        print('Deleted {} rows; {} .'.format(shape_before-csv_file_df.shape[0], value))
        return csv_file_df
    
    def delete_review(self, csv_file_df): 
        shape_before = csv_file_df.shape[0]
        og_shape =shape_before
        if self.red_list:
            red_list_df = pd.read_csv("/home/ubuntu/IMAS/Code/PyTorch/Annotation_Sets/red_list.csv", dtype=str, usecols=['label_name'])
            for value in red_list_df['label_name']:
                csv_file_df = self.delete_entries(csv_file_df, 'label_name', value)

        csv_file_df = self.delete_entries(csv_file_df, 'tag_names', 'Flagged For Review')
        #print('Not being saved to reduce space.')
        # Update the index after deleting entries 
        csv_file_df.reset_index(drop=True, inplace=True)
        self.save_csv(csv_file_df, 'review')
        return csv_file_df

    def get_norm_csv(self, csv_file_df):
        # Get list with all labels and count of each
        classes_df = csv_file_df.pivot_table(index = [self.row_name], aggfunc ='size')
        #classes_df = classes_df.index.drop_duplicates(keep='first')

        if self.sibling:
            print('Loading Sibling List')
            sib_list_df = pd.read_csv("/home/ubuntu/IMAS/Code/PyTorch/Annotation_Sets/sibling_list.csv", dtype=str, usecols=['Sibling_name'])
            
            # List without siblings, but Ecklonia 
            no_sib_classes_df = classes_df.drop(sib_list_df['Sibling_name'])
            # List with only siblings
            only_sib_classes_df = classes_df.loc[sib_list_df['Sibling_name']]

            # Normalize list without siblings and reduce to percentage defined by self.sib_factor
            norm_classes_df = no_sib_classes_df.div(no_sib_classes_df.drop(self.coi).sum()/(1-self.sib_factor))
            
            # Normalize list with only siblings and reduce to percentage defined by self.sib_factor
            only_sib_classes_df = only_sib_classes_df.div(only_sib_classes_df.sum()/(self.sib_factor))
            
            # Add both lists
            norm_classes_df = norm_classes_df.append(only_sib_classes_df)

        else:
            # Normalize all classes so that the overall number is equal to Ecklonia entries
            norm_classes_df = classes_df.div(classes_df.drop(self.coi).sum())
        
        norm_classes_df = (norm_classes_df.mul(classes_df[self.coi]*self.norm_factor)).astype(int)

        # Correct class of interest number to all entries
        norm_classes_df[self.coi] = classes_df[self.coi]

        if self.sibling:
            print('Dataset comprises of {} Sibling Entries and {} out of {} total entries'.format(
                norm_classes_df.loc[sib_list_df['Sibling_name']].sum(), 
                norm_classes_df.loc[self.coi], 
                norm_classes_df.sum()))
        return norm_classes_df, classes_df
    
    def normalize_set(self, csv_file_df):
        norm_classes_df, classes_df = self.get_norm_csv(csv_file_df)
        norm_classes_df = norm_classes_df[~norm_classes_df.index.duplicated(keep='first')]
        csv_file_df = csv_file_df.reset_index()
        np.random.seed(10)
        print('Normalizing Entries')
        counter = 0
        for label, amount in norm_classes_df.items():
            print('Processed {} out of {}: {}'.format(counter, len(norm_classes_df), label), end='\r')
            remove_n = classes_df[label] - amount
            csv_except_df = csv_file_df.loc[csv_file_df[self.row_name] == label]
            drop_indices = np.random.choice(csv_except_df.index, remove_n, replace=False)
            csv_file_df = csv_file_df.drop(drop_indices)
            counter +=1
            '''
            try:
                print('{} entries: {}'.format(label, csv_file_df.pivot_table(index = ['label_name'], aggfunc ='size')[label]))
            except:
                print('{} all {} entries deleted'.format(label, remove_n))
            '''
        print('Normalized Dataset size: {} with {} classes'.format(csv_file_df.pivot_table(index = [self.row_name], aggfunc ='size').sum(), len(csv_file_df.pivot_table(index = [self.row_name], aggfunc ='size'))))
        
        csv_file_df.reset_index(drop=True, inplace=True)
        #csv_name = self.coi + '_NMSC'
        self.save_csv(csv_file_df, self.coi.replace(' ', '_')  + '_NMSC')
        return csv_file_df

    def save_csv(self, csv_file_df, description):
        
        if self.sibling:
            path =here+'/Annotation_Sets/'+ str(len(csv_file_df.index))+'_' + str(int(self.sib_factor*100)) +'_sibling_'+ description +'_list.csv'
        elif self.norm_factor != 1:
            path =here+'/Annotation_Sets/'+ str(len(csv_file_df.index))+'_1_to_' + str(int(self.norm_factor)) + description +'_list.csv'
        else: 
            path = here+'/Annotation_Sets/'+ str(len(csv_file_df.index))+'_'+ description +'_list.csv'
        
        csv_file_df.to_csv(path,index=False)
        print('Saved {} entries with filename: {}'.format(len(csv_file_df.index), path))

    
if __name__ == "__main__":
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    
    here = os.path.dirname(os.path.abspath(__file__))
    #list_name = '/Annotation_Sets/Full_Annotation_List.csv'
    print('Loading CSV file, this may take a while.')
    list_name = '/Annotation_Sets/731_Full_Annotation_List_NMSC.csv'
    #list_name = '/Annotation_Sets/Full_Annotation_List.csv'
    csv_file_df= pd.read_csv(here + list_name, on_bad_lines='skip', low_memory=False) 
    
    print('Loaded {} entries with filename: {}'.format(len(csv_file_df.index), list_name))

    csv_file_df.columns = csv_file_df.columns.str.replace('[.]', '_', regex=True)
    # Used for NMSC dataset
    values = {"label_translated_name": 'Not of Interest'}
    csv_file_df = csv_file_df.fillna(value=values)

    data = modify_annotation_set()

    data.normalize_set(csv_file_df)
    
