import pandas as pd

# Read the CSV file
df = pd.read_csv('/home/ubuntu/Documents/IMAS/Code/PyTorch/Annotation_Sets/Final_Sets/731_Full_Annotation_List_NMSC.csv')

# Filter the campaigns that include the label defined by coi
coi_name = 'Macroalgal canopy cover'
seagrass_campaigns = df[df['label_translated_name'] == coi_name]['point_media_deployment_campaign_key'].unique()

# Print the campaigns, total entries count, coi entries count, and number of images per campaign
for campaign in seagrass_campaigns:
    total_entries_count = df[df['point_media_deployment_campaign_key'] == campaign].shape[0]
    seagrass_entries_count = df[(df['point_media_deployment_campaign_key'] == campaign) & (df['label_translated_name'] == coi_name)].shape[0]
    images_per_campaign = df[df['point_media_deployment_campaign_key'] == campaign]['point_media_path_best'].nunique()
    if (total_entries_count < 1300) and (seagrass_entries_count * 100 / total_entries_count > 20) and (total_entries_count > 1000):
        print(f"{campaign}, Entries Count: {total_entries_count}, Images: {images_per_campaign}, {int(seagrass_entries_count * 100 / total_entries_count)}%")

        # Select the rows corresponding to the current campaign
        campaign_rows = df[df['point_media_deployment_campaign_key'] == campaign]
        
        # Save the rows as a CSV file
        campaign_file_name = f"/home/ubuntu/Documents/IMAS/Code/PyTorch/Annotation_Sets/Final_Sets/Macroalgal_{campaign}.csv"
        campaign_rows.to_csv(campaign_file_name, index=False)
        
        # Remove the rows from the original dataframe
        df = df[df['point_media_deployment_campaign_key'] != campaign]

# Save the modified dataframe as a CSV file
df.to_csv('/home/ubuntu/Documents/IMAS/Code/PyTorch/Annotation_Sets/Final_Sets/731_Full_Annotation_List_NMSC_Macroalgal_modified.csv', index=False)