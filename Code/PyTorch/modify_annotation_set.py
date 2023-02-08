import csv
import os
from os import path
import pandas as pd
import urllib.request
import cv2
import numpy as np

class modify_annotation_set():
    '''
    def get_status(self, csv_file_df, here):

        # Get number of Ecklonia radiata entries
        num_eck = csv_file_df['label_name'].value_counts().loc['Ecklonia radiata']

        # Delete all Ecklonia entries from list to not obscure normalization
        no_eck_csv_df = csv_file_df.copy()
        no_eck_csv_df = no_eck_csv_df.loc[no_eck_csv_df['label_name'] != 'Ecklonia radiata']

        # count annotations without Ecklonia
        norm_data_df = no_eck_csv_df['label_name'].value_counts().rename_axis('label_name').reset_index(name='distr_num')

        og_distr_data_df = csv_file_df['label_name'].value_counts().rename_axis('label_name').reset_index(name='distr_num')
        # Delete all entries that are smaller than 0.1%
        norm_data_df = norm_data_df[~(norm_data_df['distr_num'] <= (0.001*len(no_eck_csv_df)))]

        norm_data_df['distr_num'] = norm_data_df['distr_num'] / norm_data_df['distr_num'].sum()

        return num_eck, norm_data_df, og_distr_data_df
    '''
    def __init__(self):
        self.sibling = True

    def delete_entries(self, csv_file_df, label, value):
        shape_before = csv_file_df.shape[0]
        csv_file_df = csv_file_df.loc[csv_file_df[label] != value]
        print('Deleted {} rows; {} .'.format(shape_before-csv_file_df.shape[0], value))
        return csv_file_df
    
    def delete_review(self, csv_file_df): 
        shape_before = csv_file_df.shape[0]
        og_shape =shape_before
        red_list_df = pd.read_csv("/home/ubuntu/IMAS/Code/PyTorch/Annotation_Sets/red_list.csv", dtype=str, usecols=['label_name'])
        for value in red_list_df['label_name']:
            csv_file_df = self.delete_entries(csv_file_df, 'label_name', value)
        
        csv_file_df = self.delete_entries(csv_file_df, 'tag_names', 'Flagged For Review')
        #print('Not being saved to reduce space.')
        # Update the index after deleting entries 
        csv_file_df.reset_index(drop=True, inplace=True)
        self.save_csv(csv_file_df, 'review')
        return csv_file_df
    '''
    def create_adap_set(self, csv_file_df, here, num_eck, norm_data_df, og_distr_data_df):
        sma_norm_data_df = norm_data_df.copy()
        
        sma_norm_data_df['distr_num'] = sma_norm_data_df['distr_num'] * num_eck

        norm_data_df['distr_num'] = norm_data_df['distr_num'] * len(csv_file_df.index)

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
        csv_file_df =csv_file_df.sample(frac=1)
        csv_file_df =csv_file_df.sort_values(by=['label_name'], ignore_index = True)

        # distr_data_df lists how many entries of each label are needed
        distr_data_df = distr_data_df.sort_values(by=['label_name'], ignore_index = True)

        # og_distr_data_df lists the original amount of entries per label
        og_distr_data_df = og_distr_data_df.sort_values(by=['label_name'], ignore_index = True)

        index = 0
        distr_index = 0
        print("Will be keeping {} entries, {} of those Ecklonia".format(distr_data_df['distr_num'].sum(), int(eck_df['distr_num'])))
        print('Number of entries in distr_data_df {}; number of relevant entries {}'.format(len(og_distr_data_df.index), len(distr_data_df.index)))
        csv_file_df.to_csv(here+'/Annotation_Sets/'+ str(len(csv_file_df.index))+'_REVIEWED_ANNOTATION_LIST.csv',index=False)
        
        og_index = 0
        for i in range(len(og_distr_data_df.index)):
            
            print('Index: {} looking for {}'.format(index, distr_data_df.at[distr_index, 'label_name']))
            # Get len of CSV file before operation
            len_before = len(csv_file_df.index)
            
            # Get desired number of entries
            desired_num = distr_data_df.at[distr_index,'distr_num']
            
            # Get the complete number of entries with that value
            whole_amount = int(og_distr_data_df.at[i, 'distr_num'])

            # Check whether entry is of interest, if not skip amount of index
            if og_distr_data_df.at[i,'label_name'] == distr_data_df.at[distr_index, 'label_name']:
                # Drop the entries from the last index + the desired amount until the remaining entries are deleted
                csv_file_df.drop(csv_file_df.index[(index+desired_num):(index+whole_amount)], inplace=True)

                # Update the distr_index to the next significant entry
                distr_index += 1
                # Update the index to the last saved entry
                index += desired_num

                print('{} Deleted {} of {} entries of {}. \n This is entry no {} of {} in distr_data_df.'.format(i, len_before-len(csv_file_df.index)-desired_num, len_before-len(csv_file_df.index), og_distr_data_df.at[i,'label_name'], distr_index, len(distr_data_df.index)))

            else: # The entry is not of interest delete all entries with that label
                csv_file_df.drop(csv_file_df.index[index:(index+whole_amount)], inplace=True)
                
                print('{} Deleted {} of {} entries of {}.'.format(i, len_before-len(csv_file_df.index), whole_amount, og_distr_data_df.at[i,'label_name']))

            # If all entries of distr_data_df are saved delete the rest of the entries and stop loop
            if distr_index == (len(distr_data_df.index)):
                csv_file_df.drop(csv_file_df.index[index:], inplace=True)
                # Update the index after deleting the remaining entries 
                #csv_file_df = csv_file_df.reset_index(drop=True)
                break
            
            og_index += whole_amount

            # Update the index after deleting entries 
            csv_file_df.reset_index(drop=True, inplace=True)

        print('Saving {}'.format(len(csv_file_df.index)))
        csv_file_df.to_csv(here+'/Annotation_Sets/'+ str(len(csv_file_df.index))+'_NORMALIZED_FULL_ANNOTATION_LIST.csv',index=False)
        print('Saved {} entries with filename: {}'.format(len(csv_file_df.index), (str(len(csv_file_df.index))+'_NORMALIZED_FULL_ANNOTATION_LIST.csv')))
    '''
    def get_norm_csv(self, csv_file_df):
        # Get list with all labels and count of each
        classes_df = csv_file_df.pivot_table(index = ['label_name'], aggfunc ='size')
        if self.sibling:
            sib_list_df = pd.read_csv("/home/ubuntu/IMAS/Code/PyTorch/Annotation_Sets/sibling_list.csv", dtype=str, usecols=['Sibling_name'])
            # Artificially increase number of sibling entries
            sib_classes_df = classes_df
            for label in sib_list_df['Sibling_name']:
                sib_classes_df.loc[label] = sib_classes_df.loc[label] * 5

            #!!!!!!!!!!!!!!
            # Find better way to increase siblings
            #!!!!!!!!!!!!!!



            # Normalize all classes so that the overall number is equal to Ecklonia entries, number slightly increazed by 0.95 because of download errors in later process
            norm_classes_df = sib_classes_df.div(classes_df.drop('Ecklonia radiata').sum()*0.95)
        else:
            # Normalize all classes so that the overall number is equal to Ecklonia entries, number slightly increazed by 0.95 because of download errors in later process
            norm_classes_df = classes_df.div(classes_df.drop('Ecklonia radiata').sum()*0.95)
        
        norm_classes_df = (norm_classes_df.mul(classes_df['Ecklonia radiata'])).astype(int)

        # Correct Ecklonia number to all entries
        norm_classes_df['Ecklonia radiata'] = classes_df['Ecklonia radiata']
        return norm_classes_df, classes_df
    
    def normalize_set(self, csv_file_df):
        norm_classes_df, classes_df = self.get_norm_csv(csv_file_df)
        csv_file_df = csv_file_df.reset_index()
        np.random.seed(10)
        print('Normalizing Entries')
        counter = 0
        for label, amount in norm_classes_df.items():
            print('Processed {} out of {}'.format(counter, len(norm_classes_df)), end='\r')
            remove_n = classes_df[label] - amount
            csv_except_df = csv_file_df.loc[csv_file_df['label_name'] == label]
            drop_indices = np.random.choice(csv_except_df.index, remove_n, replace=False)
            csv_file_df = csv_file_df.drop(drop_indices)
            counter +=1
            '''
            try:
                print('{} entries: {}'.format(label, csv_file_df.pivot_table(index = ['label_name'], aggfunc ='size')[label]))
            except:
                print('{} all {} entries deleted'.format(label, remove_n))
            '''
        print('Normalized Dataset size: {} with {} classes'.format(csv_file_df.pivot_table(index = ['label_name'], aggfunc ='size').sum(), len(csv_file_df.pivot_table(index = ['label_name'], aggfunc ='size'))))
        
        csv_file_df.reset_index(drop=True, inplace=True)
        self.save_csv(csv_file_df, 'normalized')
        return csv_file_df

    def save_csv(self, csv_file_df, description):
        
        if self.sibling:
            path =here+'/Annotation_Sets/'+ str(len(csv_file_df.index))+'_sibling_'+ description +'_list.csv'
        else: 
            path = here+'/Annotation_Sets/'+ str(len(csv_file_df.index))+'_'+ description +'_list.csv'
        
        csv_file_df.to_csv(path,index=False)
        print('Saved {} entries with filename: {}'.format(len(csv_file_df.index), path))

    
if __name__ == "__main__":
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    
    here = os.path.dirname(os.path.abspath(__file__))
    #list_name = '/Annotation_Sets/Full_Annotation_List.csv'
    print('Loading CSV file, this may take a while.')
    list_name = '/Annotation_Sets/1135142_review_list.csv'
    csv_file_df= pd.read_csv(here + list_name, on_bad_lines='skip', dtype={'label_name': 'str', 'tag_names': 'str'}, low_memory=False) 
    
    print('Loaded {} entries with filename: {}'.format(len(csv_file_df.index), list_name))

    csv_file_df.columns = csv_file_df.columns.str.replace('[.]', '_', regex=True)

    data = modify_annotation_set()
    
    #csv_file_df = data.delete_review(csv_file_df)
    
    data.normalize_set(csv_file_df)
    #num_eck, norm_data_df, og_distr_data_df = data.get_status(csv_file_df, here)
    
    #data.create_adap_set(csv_file_df, here, num_eck, norm_data_df, og_distr_data_df)
