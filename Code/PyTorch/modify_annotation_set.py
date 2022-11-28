import csv
import os
from os import path
import pandas as pd
import urllib.request
import cv2
import numpy as np

class modify_annotation_set():
    
    def get_status(self, CSV_FILE_DF, HERE):
        print(CSV_FILE_DF.shape)

        # Get number of Ecklonia radiata entries
        num_eck = CSV_FILE_DF['label_name'].value_counts().loc['Ecklonia radiata']
        # Get distribution of all entries
        distr_data_df = CSV_FILE_DF['label_name'].value_counts(normalize=True).to_frame()

        distr_data_df = distr_data_df.rename(columns={'': 'label_name', 'label_name':'distr_num'})

        return num_eck, distr_data_df

    def delete_review(self, CSV_FILE_DF):
        print()
    
    def create_adap_set(self, CSV_FILE_DF, HERE, num_eck, norm_data_df):
        # Create dataset containing desired distribution
        distr_data_df = norm_data_df['distr_num'] * num_eck
        distr_data_df = distr_data_df.astype({'distr_num':'int'})
        distr_data_df = distr_data_df.to_frame()
        # Set number of Ecklonia entries to amount of all entries to get 50:50
        distr_data_df[distr_data_df.columns[0]].loc['Ecklonia Radiata'] = num_eck

        distr_data_df['amount'] = 0
        print(distr_data_df.head())
        # Shuffle original dataframe
        CSV_FILE_DF =CSV_FILE_DF.sample(frac=1)
        
        '''
        # Sort both datasets
        CSV_FILE_DF.sort_values(by=['label_name'])
        #distr_data_df.sort_values(by=['index'])

        print('Start manipulation of dataset.')
        
        for index, i in enumerate(distr_data_df):
            # This gives the desired number of samples
            desired_num = distr_data_df[distr_data_df.columns[0]].loc[i]

        
        distr_data_df.to_csv(HERE+'/test2.csv', header=True, index=True, sep=' ', mode='a')
        
        '''
        counter_saved = 0
        
        
        
        for index, i in enumerate(CSV_FILE_DF['label_name']):
            # This gives the desired and current number of samples
            desired_num = distr_data_df[distr_data_df.columns[0]].loc[i]
            current_num = distr_data_df[distr_data_df.columns[1]].loc[i]
            if desired_num >= current_num:
                distr_data_df[distr_data_df.columns[1]].loc[i] += 1
                counter_saved += 1
                #print('Number of {} is {}'.format(i ,distr_data_df[distr_data_df.columns[1]].loc[i]))
                
            else:
                #print('Deleted one entry of: {}'.format(i))
                CSV_FILE_DF = CSV_FILE_DF.drop(index)
            
            if counter_saved == num_eck * 2:
                break
            
            if index % 1000 == 0:
                print('Edited {} entries and saved {}!'.format(index, counter_saved))
        
        print(CSV_FILE_DF['label_name'].value_counts(normalize=True))
        CSV_FILE_DF.to_csv(HERE+'/test2.csv', header=True, index=True, sep=' ', mode='a')



    
if __name__ == "__main__":
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    # dtype={'first_column': 'str', 'second_column': 'str'}
    HERE = os.path.dirname(os.path.abspath(__file__))
    LIST_NAME = '/Annotation_Sets/Full_Annotation_List.csv'
    print('Loading CSV file, this may take a while.')
    CSV_FILE_DF= pd.read_csv(HERE + LIST_NAME, on_bad_lines='skip')
    CSV_FILE_DF.columns = CSV_FILE_DF.columns.str.replace('[.]', '_')

    data = modify_annotation_set()
    num_eck, distr_data_df = data.get_status(CSV_FILE_DF, HERE)
    data.delete_review(CSV_FILE_DF)
    data.create_adap_set(CSV_FILE_DF, HERE, num_eck, distr_data_df)
