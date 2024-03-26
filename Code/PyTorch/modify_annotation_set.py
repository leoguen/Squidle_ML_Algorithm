import re
import pandas as pd
import numpy as np
import argparse

class modify_annotation_set():

    def __init__(self, args):
        self.sibling = args.sibling
        self.sibling_list_path = args.sibling_list_path
        self.red_list = args.red_list
        self.neighbour = args.neighbour
        self.sib_factor = args.sib_factor
        self.coi = args.coi
        self.defined_name = args.defined_name
        self.col_name = args.col_name
        self.norm_factor = args.norm_factor
        self.save_path = args.save_path
        self.red_list_path = args.red_list_path
        self.annotationset_path = args.annotationset_path

    def replace_lineage(self, csv_file_df):
        """
        Finds lineages that include the target lineage but also go more into depth and replace that with the lineage depth that is required
        """
        count = (csv_file_df[self.col_name] == self.coi).sum()
        print("Number of {} Entries: {} After replace_lineage".format(self.coi, count))
        # Create a boolean mask to identify rows that contain the substring
        mask = csv_file_df[self.col_name].str.contains(self.coi, case=False, na=False, regex=False)

        # Replace the values in the specified column with self.coi where the mask is True
        csv_file_df.loc[mask, self.col_name] = self.coi

        count = (csv_file_df[self.col_name] == self.coi).sum()
        print("Number of {} Entries: {} After replace_lineage".format(self.coi, count))

        return csv_file_df


    def delete_entries(self, csv_file_df, label, value):
        shape_before = csv_file_df.shape[0]
        if value == "NaN":
            csv_file_df = csv_file_df.dropna(subset=[label])
        else:
            csv_file_df = csv_file_df.loc[csv_file_df[label] != value]
        
        if label == "point_x":
            if value == "nan":
                csv_file_df = csv_file_df.dropna(subset=[label])

        print('Deleted {} rows; {} .'.format(shape_before-csv_file_df.shape[0], value))
        return csv_file_df
    
    def delete_review(self, csv_file_df): 

        if self.red_list:
            # Delete everything that is higher hierarchy than self.coi
            parts = self.coi.split(" > ")

            # Initialize an empty list to store the results
            results = []

            # Iterate through the parts in reverse order
            for i in range(len(parts), 0, -1):
                # Join the parts back together up to the current index
                result = " > ".join(parts[:i-1])
                results.append(result)

            # Delete according to the lineage names
            for idx, lineage_result in enumerate(results):
                csv_file_df = self.delete_entries(csv_file_df, 'label_translated_lineage_names', lineage_result)
        # If a redlist path is given
            if self.red_list_path != "":
                red_list_df = pd.read_csv(self.red_list_path, dtype=str, usecols=[self.col_name])
                for value in red_list_df[self.col_name]:
                    csv_file_df = self.delete_entries(csv_file_df, self.col_name, value)

        csv_file_df = self.delete_entries(csv_file_df, 'tag_names', 'Flagged For Review')
        if 'translated' in self.col_name:
            csv_file_df = self.delete_entries(csv_file_df, 'label_translated_name', 'NaN')
            csv_file_df = self.delete_entries(csv_file_df, 'point_x', 'nan')

        #print('Not being saved to reduce space.')
        # Update the index after deleting entries 
        csv_file_df.reset_index(drop=True, inplace=True)
        #self.save_csv(csv_file_df, 'review')
        return csv_file_df

    def get_norm_csv(self, csv_file_df):
        # Get list with all labels and count of each
        classes_df = csv_file_df.pivot_table(index = [self.col_name], aggfunc ='size')
        #classes_df = classes_df.index.drop_duplicates(keep='first')
        if self.neighbour:
            print('self.neighbour is set to True')
            # Get all COI rows in csv_file_df
            only_coi_df = csv_file_df.loc[csv_file_df[self.col_name] == self.coi]

            print('Number of Entries {} in relation to {} Images'.format(len(csv_file_df[self.col_name]),len(csv_file_df.drop_duplicates(subset=['point_media_path_best'])[self.col_name])))

            print('Number of COI entries {}'.format(len(only_coi_df)))
            # Get list of image paths without duplicates
            coi_no_duplic_path = only_coi_df.drop_duplicates(subset=['point_media_path_best'])
            print('Number of COI images no duplicate {}'.format(len(coi_no_duplic_path)))
            
            # Get all annotations that belong to that image
            addi_anno_coi = csv_file_df.loc[csv_file_df['point_media_path_best'].isin(coi_no_duplic_path.loc[:,'point_media_path_best'])]
            print('Number of COI Image related entries {}'.format(len(addi_anno_coi)))

            # Drop the class of interest
            addi_anno_coi = addi_anno_coi.drop(addi_anno_coi[addi_anno_coi[self.col_name] == self.coi].index)
            print('Number of COI Image related entries without COI entries {}'.format(len(addi_anno_coi)))
            # Get all the labels and amount of entries per label in these additional annotations
            labels_addi = addi_anno_coi.pivot_table(index = [self.col_name], aggfunc ='size')
            
            # Normalize all classes so that the overall number is equal to Ecklonia entries
            norm_classes_df = classes_df.div(classes_df.drop(self.coi).sum())
            mul_factor = classes_df[self.coi]*self.norm_factor
            norm_classes_df = (norm_classes_df.mul(mul_factor)).astype(int)
            
            # Substract the amount of labels that are already given in lables_addi
            norm_classes_df = norm_classes_df.sub(labels_addi, fill_value=0)

            # Correct class of interest number to all entries
            norm_classes_df[self.coi] = classes_df[self.coi]

            # If value negative make it 0
            norm_classes_df = norm_classes_df.clip(lower=0)
            return norm_classes_df, classes_df, addi_anno_coi
            
        if self.sibling:
            print('Loading Sibling List')
            sib_list_df = pd.read_csv(self.sibling_list_path, dtype=str, usecols=['Sibling_name'])
            
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
            norm_classes_df = (norm_classes_df.mul(classes_df[self.coi]*self.norm_factor)).astype(int)

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

        if self.neighbour: 
            norm_classes_df, classes_df, addi_anno_coi = self.get_norm_csv(csv_file_df)
            # drop the additional annotations from csv_file_df
            cond = csv_file_df['point_id'].isin(addi_anno_coi['point_id'])
            csv_file_df = csv_file_df.drop(csv_file_df[cond].index)
            # Get the classes from the residual testset
            classes_df = csv_file_df.pivot_table(index = [self.col_name], aggfunc ='size')
            # Throw out all norm_calsses_df entries that are not in the excpet anymore
            cond = ~norm_classes_df.index.isin(classes_df.index)
            norm_classes_df = norm_classes_df.drop(norm_classes_df[cond].index)

        else:
            norm_classes_df, classes_df = self.get_norm_csv(csv_file_df)
        
        norm_classes_df = norm_classes_df[~norm_classes_df.index.duplicated(keep='first')]
        csv_file_df = csv_file_df.reset_index()
        np.random.seed(10)
        print('Normalizing Entries')
        counter = 0
        for label, amount in norm_classes_df.items():
            print('Processed {} out of {}: {}'.format(counter, len(norm_classes_df), label), end='\r')
            
            remove_n = classes_df[label] - int(amount)
            if remove_n < 0:
                remove_n = 0
            csv_except_df = csv_file_df.loc[csv_file_df[self.col_name] == label]
            drop_indices = np.random.choice(csv_except_df.index, remove_n, replace=False)
            csv_file_df = csv_file_df.drop(drop_indices)
            counter +=1

        if self.neighbour:
            #add addi_anno_coi to csv_file_df
            csv_file_df = pd.concat([csv_file_df, addi_anno_coi], ignore_index=True)
            csv_file_df = csv_file_df.drop_duplicates()
        print('Normalized Dataset size: {} with {} classes'.format(csv_file_df.pivot_table(index = [self.col_name], aggfunc ='size').sum(), len(csv_file_df.pivot_table(index = [self.col_name], aggfunc ='size'))))
        csv_file_df.reset_index(drop=True, inplace=True)
        #csv_name = self.coi + '_NMSC'
        file_name = self.clean_and_extract_filename(self.coi)
        self.save_csv(csv_file_df, file_name  + self.defined_name)
        return csv_file_df

    def save_csv(self, csv_file_df, description):
        if self.red_list: 
            description = description + '_red_listed'
        if self.sibling:
            path = self.save_path+ str(len(csv_file_df.index))+'_' + str(int(self.sib_factor*100)) +'_sibling_'+ description +'.csv'
        elif self.norm_factor != 1:
            path =self.save_path + str(len(csv_file_df.index))+'_1_to_' + str(int(self.norm_factor)) + description +'.csv'
        elif self.neighbour:
            path = self.save_path + str(len(csv_file_df.index))+'_neighbour_' + description +'.csv'
        else: 
            path = self.save_path + str(len(csv_file_df.index))+'_'+ description +'.csv'
        
        csv_file_df.to_csv(path,index=False)
        print('Saved {} entries with filename: {}'.format(len(csv_file_df.index), path))

    def clean_and_extract_filename(self, input_string):
        # Split the input string by ">"
        parts = input_string.split(">")
        
        # Take the last part of the split as the filename
        filename = parts[-1].strip()
        
        # Remove special characters that are not allowed in filenames
        cleaned_filename = re.sub(r'[\/:*?"<>|]', '_', filename)
        
        return cleaned_filename
    
def get_args():
    parser = argparse.ArgumentParser(description="Modify Annotation Set")
    parser.add_argument('--sibling', action='store_true', help="Enable sibling processing (default: False)")
    parser.add_argument('--sibling_list_path', default="./Annotation_Sets/sibling_list.csv", help="Path to the sibling list CSV file (default: '/home/ubuntu/IMAS/Code/PyTorch/Annotation_Sets/sibling_list.csv')")
    parser.add_argument('--red_list', action='store_true', help="Specifies whether certain entries should be marked as red-listed. If working with lineage_names it is recommended to set this true even if you do not want to supply a red_list file. If set to True (which is default) the csv file will also be browsed for higher hierarchy lenage_names that might include your coi but would be handled as others category. E.g. when using the lineage *Physical > Substrate > Unconsolidated (soft) > Sand / mud (<2mm)*, it would delete all *Physical > Substrate > Unconsolidated (soft)*, *Physical > Substrate* and *Physical*, as those might not exclusively contain our coi.")
    parser.add_argument('--red_list_path', type=str, default="", help="Determines the path where the red list is saved")
    parser.add_argument('--neighbour', action='store_false', help="Disable neighbour processing (default: True)")
    parser.add_argument('--sib_factor', type=float, default=0.3, help="Sibling factor (default: 0.3)")
    parser.add_argument('--coi', default='Physical > Substrate > Unconsolidated (soft) > Sand / mud (<2mm)', help="Class of Interest (default: 'Physical > Substrate > Unconsolidated (soft) > Sand / mud (<2mm)')")
    parser.add_argument('--defined_name', default="", help="Additional string for the filename (default: '')")
    parser.add_argument('--col_name', default='label_translated_lineage_names', help="Column name (default: 'label_translated_lineage_names')")
    parser.add_argument('--norm_factor', type=int, default=1, help="Normalization factor (default: 1)")
    parser.add_argument('--save_path', default="./Annotationsets/", help="Save path (default: './Annotationsets/')")
    parser.add_argument('--annotationset_path', default='./Annotationsets/20240325_Full_Annotation_List.csv', help="Path where your annotation set is saved.")
    return parser.parse_args()

if __name__ == "__main__":
    pd.set_option("display.max_rows", 10, "display.max_columns", 10)
    
    print('Loading CSV file, this may take a while.')
    args = get_args()
    data = modify_annotation_set(args)
    
    csv_file_df= pd.read_csv(args.annotationset_path, on_bad_lines='skip', low_memory=False) 
    
    print('Loaded {} entries with filename: {}'.format(len(csv_file_df.index), args.annotationset_path))

    csv_file_df.columns = csv_file_df.columns.str.replace('[.]', '_', regex=True) #replace dots with underscores

    # Used for NMSC dataset
    #values = {"label_translated_name": 'Not of Interest'}
    #csv_file_df = csv_file_df.fillna(value=values)
    
    

    # Deletes entries that should not be used (e.g. "Flagged for Review") 
    csv_file_df = data.delete_review(csv_file_df)

    # Replace the more in depth lineages with the target lineage if required
    csv_file_df = data.replace_lineage(csv_file_df)

    data.normalize_set(csv_file_df)