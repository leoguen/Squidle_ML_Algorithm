import csv
import os
from os import path
import pandas as pd
import urllib.request
import cv2
import numpy as np

class modify_annotation_set():
    
    def get_status(self, CSV_FILE_DF, HERE):

        # Get number of Ecklonia radiata entries
        num_eck = CSV_FILE_DF['label_name'].value_counts().loc['Ecklonia radiata']

        # Delete all Ecklnia entries from list to not obscure normalization
        NO_ECK_CSV_DF = CSV_FILE_DF.loc[CSV_FILE_DF['label_name'] != 'Ecklonia radiata']

        # Normalize data without Ecklonia
        norm_data_df = NO_ECK_CSV_DF['label_uuid'].value_counts(normalize=True).rename_axis('label_uuid').reset_index(name='distr_num')
        return num_eck, norm_data_df, 

    def delete_review(self, CSV_FILE_DF):
        
        shape_before = CSV_FILE_DF.shape[0]
        CSV_FILE_DF = CSV_FILE_DF.loc[CSV_FILE_DF['tag_names'] != 'Flagged For Review']

        print('Deleted {} rows; "Flagged For Review" .'.format(shape_before-CSV_FILE_DF.shape[0]))
        shape_before = CSV_FILE_DF.shape[0]

        CSV_FILE_DF = CSV_FILE_DF.loc[CSV_FILE_DF['label_uuid'] != '2a00a85e-6675-4d69-ab4f-197992dcead7']
        
        print('Deleted {} rows; Unscorable.'.format(shape_before-CSV_FILE_DF.shape[0]))
        shape_before = CSV_FILE_DF.shape[0]
        
        CSV_FILE_DF = CSV_FILE_DF.loc[CSV_FILE_DF['point_media_key'].str.startswith('https')==False]
        
        print('Deleted {} rows; wrong URL.'.format(shape_before-CSV_FILE_DF.shape[0]))
        shape_before = CSV_FILE_DF.shape[0]

        CSV_FILE_DF = CSV_FILE_DF[CSV_FILE_DF['point_pose_alt'].apply(lambda x: str(x).replace('.', '', 1).isdigit())]
        print('Deleted {} rows; missplaced rows.'.format(shape_before-CSV_FILE_DF.shape[0]))
        shape_before = CSV_FILE_DF.shape[0]
        return CSV_FILE_DF
    
    def create_adap_set(self, CSV_FILE_DF, HERE, num_eck, norm_data_df):

        # Delete all entries that are smaller than 0.1%
        sma_norm_data_df = norm_data_df[~(norm_data_df['distr_num'] <= 0.001)] 

        # Create dataset containing desired distribution without Ecklonia
        sma_norm_data_df.loc[:,'distr_num'] = sma_norm_data_df.loc[:,'distr_num'] * num_eck

        norm_data_df['distr_num'] = norm_data_df['distr_num'] * len(CSV_FILE_DF.index)

        # distr_data_df contains the distribution which should be achieved
        distr_data_df = sma_norm_data_df
        distr_data_df = distr_data_df.astype({'distr_num':'int'})

        # og_distr_data_df contains the distribution from the original .csv file
        og_distr_data_df = norm_data_df
        
        # Create dataframe with according amount of Ecklonia data
        # Not taking num_eck here because some percent are lost by deleting <= 0.1%
        eck_df = pd.DataFrame({'label_uuid': ['5b3ca3b9-20c0-4a7f-a08e-14ebd493f9cf'],
        'distr_num' : [int(distr_data_df['distr_num'].sum())]})

        # Join Ecklonia Dataframe
        distr_data_df = pd.concat([distr_data_df, eck_df], ignore_index=True)
        
        # Shuffle original dataframe
        CSV_FILE_DF =CSV_FILE_DF.sample(frac=1)
        CSV_FILE_DF =CSV_FILE_DF.sort_values(by=['label_uuid'], ignore_index = True)

        # distr_data_df lists how many entries of each label are needed
        distr_data_df = distr_data_df.sort_values(by=['label_uuid'], ignore_index = True)

        # og_distr_data_df lists the original amount of entries per label
        og_distr_data_df = og_distr_data_df.sort_values(by=['label_uuid'], ignore_index = True)

        index = 0
        distr_index = 0
        print("Will be keeping {} entries, {} of those Ecklonia".format(distr_data_df['distr_num'].sum(), int(eck_df['distr_num'])))
        
        
        for i in range(len(og_distr_data_df.index)):
            #get desired number of entries
            desired_num = distr_data_df.at[distr_index,'distr_num']
            
            # Get the complete number of entries with that value
            whole_amount = int(og_distr_data_df.at[i, 'distr_num'])

            # Check whether entry is of interest, if not skip amount of index
            if og_distr_data_df.at[i,'label_uuid'] == distr_data_df.at[distr_index, 'label_uuid']:
                # Drop the entries from the last index + the desired amount until the remaining entries are deleted
                CSV_FILE_DF = CSV_FILE_DF.drop(CSV_FILE_DF.index[(index+desired_num):whole_amount])

                # Update the distr_index to the next significant entry
                distr_index += 1
                # Update the index to the last saved entry
                index += desired_num

                print('{} Deleted {} and kept {} entries of {}. This is entry no {} of {} in distr_data_df.'.format(i, whole_amount, desired_num, og_distr_data_df.at[i,'label_uuid'], distr_index, len(distr_data_df.index)))
            else: # The entry is not of interest delete all entries with that label
                CSV_FILE_DF = CSV_FILE_DF.drop(CSV_FILE_DF.index[(index):whole_amount])
                print('{} Deleted {} entries of {}.'.format(i, whole_amount, og_distr_data_df.at[i,'label_uuid']))

            # If all entries of distr_data_df are saved delete the rest of the entries and stop loop
            if distr_index == (len(distr_data_df.index)):
                CSV_FILE_DF = CSV_FILE_DF.drop(CSV_FILE_DF.index[index:])
                break
            # Update the index after deleting entries 
            CSV_FILE_DF = CSV_FILE_DF.reset_index(drop=True)

            

        CSV_FILE_DF.to_csv(HERE+'/Annotation_Sets/'+ str(len(CSV_FILE_DF.index))+'_NORMALIZED_FULL_ANNOTATION_LIST.csv',index=False)
        print('Saved {} entries with filename: '.format(distr_data_df['distr_num'].sum(), str(len(CSV_FILE_DF.index))+'_NORMALIZED_FULL_ANNOTATION_LIST.csv'))




    
if __name__ == "__main__":
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    # dtype={'first_column': 'str', 'second_column': 'str'}
    HERE = os.path.dirname(os.path.abspath(__file__))
    LIST_NAME = '/Annotation_Sets/Full_Annotation_List.csv'
    print('Loading CSV file, this may take a while.')
    CSV_FILE_DF= pd.read_csv(HERE + LIST_NAME, on_bad_lines='skip', dtype={'label_name': 'str', 'tag_names': 'str'}) 
    '''
    dtype=
        {'label_id': 'int', 
        'label_name': 'str',
        'label_uuid': 'str',
        'point.data.user_created': 'str',
        'point.id': 'int',
        'point.media.id': 'str',
        'point.media.key': 'str',
        'point.media.path_best': 'str',
        'point.pose.alt': 'float',
        'point.pose.dep': 'float',
        'point.pose.lat': 'float',
        'point.pose.lon': 'float',
        'point.pose.timespamp': 'str',
        'point_x': 'float',
        'point_y': 'float',
        'tag_names': 'str',
        })
    '''
    CSV_FILE_DF.columns = CSV_FILE_DF.columns.str.replace('[.]', '_', regex=True)
    
    #CSV_FILE_DF.to_csv(HERE+'/SKIPPED_FULL_ANNOTATION_LIST.csv',index=False)
    data = modify_annotation_set()
    
    CSV_FILE_DF = data.delete_review(CSV_FILE_DF)
    num_eck, norm_data_df = data.get_status(CSV_FILE_DF, HERE)
    
    data.create_adap_set(CSV_FILE_DF, HERE, num_eck, norm_data_df)
