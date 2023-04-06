import pandas as pd

# Read the CSV file
df = pd.read_csv('/home/ubuntu/Documents/IMAS/Code/PyTorch/Annotation_Sets/Final_Sets/680037_1_to_10_Ecklonia_radiata.csv')
df1 = df

# Filter the campaigns that include the label defined by coi
row_name = 'label_name' #'label_translated_name'
coi_name = 'Ecklonia radiata'
media_path = 'point_media_path_best'
campaign_key = 'point_media_deployment_campaign_key'
seagrass_campaigns = df[df[row_name] == coi_name][campaign_key].unique()

# Print the campaigns, total entries count, coi entries count, and number of images per campaign
for campaign in seagrass_campaigns:
    total_entries_count = df[df[campaign_key] == campaign].shape[0]
    seagrass_entries_count = df[(df[campaign_key] == campaign) & (df[row_name] == coi_name)].shape[0]
    images_per_campaign = df[df[campaign_key] == campaign][media_path].nunique()
    if (campaign == 'RLS_Gulf St Vincent_2012') or (campaign == 'RLS_Jervis Bay Marine Park_2015')or (campaign == 'RLS_Port Phillip Heads_2010'):
        print(f"{campaign}, Entries Count: {total_entries_count}, Images: {images_per_campaign}, {int(seagrass_entries_count * 100 / total_entries_count)}%")

        # Select the rows corresponding to the current campaign
        campaign_rows = df[df[campaign_key] == campaign]
        
        # Save the rows as a CSV file
        campaign_file_name = f"/home/ubuntu/Documents/IMAS/Code/PyTorch/Annotation_Sets/Final_Sets/Ecklonia_{campaign}.csv"
        #campaign_rows.to_csv(campaign_file_name, index=False)
        
        # Remove the rows from the original dataframe
        df = df[df[campaign_key] != campaign]

# Save the modified dataframe as a CSV file
df.to_csv('/home/ubuntu/Documents/IMAS/Code/PyTorch/Annotation_Sets/Final_Sets/680037_1_to_10_Ecklonia_radiata_except.csv', index=False)

# Step 2: Read CSV files and store in separate data frames


# Step 3: Merge the data frames and include a column indicating the source
merged = pd.merge(df1, df, on='point_id', how='outer', indicator=True)

# Step 4: Filter the merged data frame to include only the rows where the source is 'left_only' or 'right_only'
diff_entries = merged.loc[(merged['_merge'] != 'both')]

# Step 5: Count the number of rows in the filtered data frame to get the number of different entries
num_diff_entries = len(diff_entries)

# Print the number of different entries
print(f'The number of different entries between the two CSV files is {num_diff_entries}')
