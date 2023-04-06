import pandas as pd

# Read the CSV file
df = pd.read_csv('/home/ubuntu/Documents/IMAS/Code/PyTorch/Annotation_Sets/Final_Sets/737_Full_Annotation_List.csv')

# Filter the campaigns that include the label defined by coi
row_name = 'label.name' #'label_translated_name'
coi_name = 'Ecklonia radiata'
media_path = 'point.media.path_best'
campaign_key = 'point.media.deployment.campaign.key'
seagrass_campaigns = df[df[row_name] == coi_name][campaign_key].unique()

# Print the campaigns, total entries count, coi entries count, and number of images per campaign
for campaign in seagrass_campaigns:
    total_entries_count = df[df[campaign_key] == campaign].shape[0]
    seagrass_entries_count = df[(df[campaign_key] == campaign) & (df[row_name] == coi_name)].shape[0]
    images_per_campaign = df[df[campaign_key] == campaign][media_path].nunique()
    if (total_entries_count < 1500) and (seagrass_entries_count * 100 / total_entries_count > 15) and (total_entries_count > 1000):
        print(f"{campaign}, Entries Count: {total_entries_count}, Images: {images_per_campaign}, {int(seagrass_entries_count * 100 / total_entries_count)}%")

        # Select the rows corresponding to the current campaign
        campaign_rows = df[df[campaign_key] == campaign]
        
        # Save the rows as a CSV file
        campaign_file_name = f"/home/ubuntu/Documents/IMAS/Code/PyTorch/Annotation_Sets/Final_Sets/Ecklonia_{campaign}.csv"
        campaign_rows.to_csv(campaign_file_name, index=False)
        
        # Remove the rows from the original dataframe
        df = df[df[campaign_key] != campaign]

# Save the modified dataframe as a CSV file
df.to_csv('/home/ubuntu/Documents/IMAS/Code/PyTorch/Annotation_Sets/Final_Sets/731_Full_Annotation_List_NMSC_Ecklonia_modified.csv', index=False)