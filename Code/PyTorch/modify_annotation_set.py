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

        # Delete all Ecklonia entries from list to not obscure normalization
        NO_ECK_CSV_DF = CSV_FILE_DF.copy()
        NO_ECK_CSV_DF = NO_ECK_CSV_DF.loc[NO_ECK_CSV_DF['label_name'] != 'Ecklonia radiata']

        # count annotations without Ecklonia
        norm_data_df = NO_ECK_CSV_DF['label_name'].value_counts().rename_axis('label_name').reset_index(name='distr_num')

        og_distr_data_df = CSV_FILE_DF['label_name'].value_counts().rename_axis('label_name').reset_index(name='distr_num')
        # Delete all entries that are smaller than 0.1%
        norm_data_df = norm_data_df[~(norm_data_df['distr_num'] <= (0.001*len(NO_ECK_CSV_DF)))]

        norm_data_df['distr_num'] = norm_data_df['distr_num'] / norm_data_df['distr_num'].sum()

        return num_eck, norm_data_df, og_distr_data_df

    def delete_review(self, CSV_FILE_DF): 
        shape_before = CSV_FILE_DF.shape[0]
        og_shape =shape_before
        CSV_FILE_DF = CSV_FILE_DF.loc[CSV_FILE_DF['tag_names'] != 'Flagged For Review']

        print('Deleted {} rows; "Flagged For Review" .'.format(shape_before-CSV_FILE_DF.shape[0]))
        shape_before = CSV_FILE_DF.shape[0]

        CSV_FILE_DF = CSV_FILE_DF.loc[CSV_FILE_DF['label_uuid'] != '2a00a85e-6675-4d69-ab4f-197992dcead7'] # Unscorable
        CSV_FILE_DF = CSV_FILE_DF.loc[CSV_FILE_DF['label_uuid'] != 'a00b6b37-0dc0-4bd5-893b-4ae1788fa96d'] # Unscorable
        CSV_FILE_DF = CSV_FILE_DF.loc[CSV_FILE_DF['label_uuid'] != '79a9095f-115c-45a0-9423-47d1e92acb7f'] # Unknown
        CSV_FILE_DF = CSV_FILE_DF.loc[CSV_FILE_DF['label_uuid'] != '8a4c6e1b-66a1-48a2-b1d8-a36d6af483ef'] # Unknown
        
        print('Deleted {} rows; Unscorable.'.format(shape_before-CSV_FILE_DF.shape[0]))
        shape_before = CSV_FILE_DF.shape[0]
        
        CSV_FILE_DF = CSV_FILE_DF.loc[CSV_FILE_DF['point_media_path_best'].str.startswith('https')==True]

        print('Deleted {} rows; wrong URL.'.format(shape_before-CSV_FILE_DF.shape[0]))
        shape_before = CSV_FILE_DF.shape[0]
        
        CSV_FILE_DF['point_y'].replace('', np.nan, inplace=True)
        CSV_FILE_DF.dropna(subset=['point_y'], inplace=True)

        print('Deleted {} rows; empty values for point_y.'.format(shape_before-CSV_FILE_DF.shape[0]))
        shape_before = CSV_FILE_DF.shape[0]

        #CSV_FILE_DF = CSV_FILE_DF[CSV_FILE_DF['point_pose_alt'].apply(lambda x: str(x).replace('.', '', 1).isdigit())]
        #print('Deleted {} rows; missplaced rows.'.format(shape_before-CSV_FILE_DF.shape[0]))
        #shape_before = CSV_FILE_DF.shape[0]
        
        print('The csv file consists of {} entries, {} were deleted. \nThe file is being saved as {}.'.format(CSV_FILE_DF.shape[0], (og_shape-CSV_FILE_DF.shape[0]), str(len(CSV_FILE_DF.index))+'_REVIEWED_ANNOTATION_LIST.csv'))

        #print('Not being saved to reduce space.')
        # Update the index after deleting entries 
        CSV_FILE_DF.reset_index(drop=True, inplace=True)
        CSV_FILE_DF.to_csv(HERE+'/Annotation_Sets/'+ str(len(CSV_FILE_DF.index))+'_REVIEWED_ANNOTATION_LIST.csv',index=False)
        
        return CSV_FILE_DF
    
    def create_adap_set(self, CSV_FILE_DF, HERE, num_eck, norm_data_df, og_distr_data_df):
        sma_norm_data_df = norm_data_df.copy()
        
        sma_norm_data_df['distr_num'] = sma_norm_data_df['distr_num'] * num_eck

        norm_data_df['distr_num'] = norm_data_df['distr_num'] * len(CSV_FILE_DF.index)

        # distr_data_df contains the distribution which should be achieved
        distr_data_df = sma_norm_data_df
        distr_data_df = distr_data_df.astype({'distr_num':'int'})
        
        # Create dataframe with according amount of Ecklonia data
        # Not taking num_eck here because some percent are lost by deleting <= 0.1%
        eck_df = pd.DataFrame({'label_name': ['Ecklonia radiata'],
        # Ecklonia radiata
        'distr_num' : [int(distr_data_df['distr_num'].sum())]})

        # Join Ecklonia Dataframe
        distr_data_df = pd.concat([distr_data_df, eck_df], ignore_index=True)

        # Shuffle original dataframe
        CSV_FILE_DF =CSV_FILE_DF.sample(frac=1)
        CSV_FILE_DF =CSV_FILE_DF.sort_values(by=['label_name'], ignore_index = True)

        # distr_data_df lists how many entries of each label are needed
        distr_data_df = distr_data_df.sort_values(by=['label_name'], ignore_index = True)

        # og_distr_data_df lists the original amount of entries per label
        og_distr_data_df = og_distr_data_df.sort_values(by=['label_name'], ignore_index = True)

        index = 0
        distr_index = 0
        print("Will be keeping {} entries, {} of those Ecklonia".format(distr_data_df['distr_num'].sum(), int(eck_df['distr_num'])))
        print('Number of entries in distr_data_df {}; number of relevant entries {}'.format(len(og_distr_data_df.index), len(distr_data_df.index)))
        CSV_FILE_DF.to_csv(HERE+'/Annotation_Sets/'+ str(len(CSV_FILE_DF.index))+'_REVIEWED_ANNOTATION_LIST.csv',index=False)
        
        og_index = 0
        for i in range(len(og_distr_data_df.index)):
            
            print('Index: {} looking for {}'.format(index, distr_data_df.at[distr_index, 'label_name']))
            # Get len of CSV file before operation
            len_before = len(CSV_FILE_DF.index)
            
            # Get desired number of entries
            desired_num = distr_data_df.at[distr_index,'distr_num']
            
            # Get the complete number of entries with that value
            whole_amount = int(og_distr_data_df.at[i, 'distr_num'])

            # Check whether entry is of interest, if not skip amount of index
            if og_distr_data_df.at[i,'label_name'] == distr_data_df.at[distr_index, 'label_name']:
                # Drop the entries from the last index + the desired amount until the remaining entries are deleted
                CSV_FILE_DF.drop(CSV_FILE_DF.index[(index+desired_num):(index+whole_amount)], inplace=True)

                # Update the distr_index to the next significant entry
                distr_index += 1
                # Update the index to the last saved entry
                index += desired_num

                print('{} Deleted {} of {} entries of {}. \n This is entry no {} of {} in distr_data_df.'.format(i, len_before-len(CSV_FILE_DF.index)-desired_num, len_before-len(CSV_FILE_DF.index), og_distr_data_df.at[i,'label_name'], distr_index, len(distr_data_df.index)))

            else: # The entry is not of interest delete all entries with that label
                CSV_FILE_DF.drop(CSV_FILE_DF.index[index:(index+whole_amount)], inplace=True)
                
                print('{} Deleted {} of {} entries of {}.'.format(i, len_before-len(CSV_FILE_DF.index), whole_amount, og_distr_data_df.at[i,'label_name']))

            # If all entries of distr_data_df are saved delete the rest of the entries and stop loop
            if distr_index == (len(distr_data_df.index)):
                CSV_FILE_DF.drop(CSV_FILE_DF.index[index:], inplace=True)
                # Update the index after deleting the remaining entries 
                #CSV_FILE_DF = CSV_FILE_DF.reset_index(drop=True)
                break
            
            og_index += whole_amount

            # Update the index after deleting entries 
            CSV_FILE_DF.reset_index(drop=True, inplace=True)

        print('Saving {}'.format(len(CSV_FILE_DF.index)))
        CSV_FILE_DF.to_csv(HERE+'/Annotation_Sets/'+ str(len(CSV_FILE_DF.index))+'_NORMALIZED_FULL_ANNOTATION_LIST.csv',index=False)
        print('Saved {} entries with filename: {}'.format(len(CSV_FILE_DF.index), (str(len(CSV_FILE_DF.index))+'_NORMALIZED_FULL_ANNOTATION_LIST.csv')))




    
if __name__ == "__main__":
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    # dtype={'first_column': 'str', 'second_column': 'str'}
    HERE = os.path.dirname(os.path.abspath(__file__))
    LIST_NAME = '/Annotation_Sets/Full_Annotation_List.csv'
    print('Loading CSV file, this may take a while.')
    CSV_FILE_DF= pd.read_csv(HERE + LIST_NAME, on_bad_lines='skip', dtype={'label_name': 'str', 'tag_names': 'str'}, low_memory=False) 
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

    data = modify_annotation_set()
    
    CSV_FILE_DF = data.delete_review(CSV_FILE_DF)
    num_eck, norm_data_df, og_distr_data_df = data.get_status(CSV_FILE_DF, HERE)
    data.create_adap_set(CSV_FILE_DF, HERE, num_eck, norm_data_df, og_distr_data_df)
