import pandas as pd

# Load the first CSV file
df1 = pd.read_csv('/home/ubuntu/Documents/IMAS/Code/PyTorch/Annotation_Sets/14960_Seagrass_cover_NMSC_list.csv')

# Load the second CSV file
df2 = pd.read_csv('/home/ubuntu/Documents/IMAS/Code/PyTorch/Annotation_Sets/14961_Seagrass_cover_NMSC_list.csv')
# Count the number of values in df1['point_id'] that are also in df2['point_id']
common_points = len(set(df1['point_id']).intersection(set(df2['point_id'])))

# Print the number of common values
print(f"There are {common_points} values in df1['point_id'] that are also in df2['point_id'].")