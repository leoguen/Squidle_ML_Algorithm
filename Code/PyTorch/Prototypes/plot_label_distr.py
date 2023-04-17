import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read CSV files into pandas dataframes
df1 = pd.read_csv('/home/leo/Documents/IMAS/Code/PyTorch/Annotation_Sets/Paper_Data/1_to_1_Ecklonia_radiata.csv', header=None, names=['label', 'counter'])
df2 = pd.read_csv('/home/leo/Documents/IMAS/Code/PyTorch/Annotation_Sets/Paper_Data/1_to_10_Ecklonia_radiata.csv', header=None, names=['label', 'counter'])
df3 = pd.read_csv('/home/leo/Documents/IMAS/Code/PyTorch/Annotation_Sets/Paper_Data/1_to_1_neigh_Ecklonia_radiata.csv', header=None, names=['label', 'counter'])

# Sort the labels by count in descending order and select the top 8
df1_top8 = df1.sort_values(by='counter', ascending=False).head(8)
df2_top8 = df2.sort_values(by='counter', ascending=False).head(8)
df3_top8 = df3.sort_values(by='counter', ascending=False).head(8)

df1_top8 = df1_top8.reset_index(drop=True)
df2_top8 = df2_top8.reset_index(drop=True)
df3_top8 = df3_top8.reset_index(drop=True)

# Calculate the total count and percent for the top 8 labels
for top_8 in [df1_top8, df2_top8, df3_top8]:
    total_count = top_8['counter'].sum()
    top_8['percent'] = top_8['counter'] / total_count * 100
    print(top_8['label'])
    top_8.loc[top_8['label'] == 'Turfing algae (<2 cm high algal/sediment mat on rock)', 'label'] = 'Turfing algae'
# Define the colors for each datafram

color_palette = ['#1f77b4', '#FF4136', '#2ECC40', '#FFDC00', '#0074D9', '#FF851B', '#7FDBFF', '#B10DC9', '#3D9970', '#F012BE', '#8E44AD', '#27AE60']
for top_8 in [df1_top8, df2_top8, df3_top8]:
    top_8['color'] = top_8['label']
    top_8.loc[top_8['color'] == 'Unconsolidated (soft)', 'color'] = color_palette[0]
    top_8.loc[top_8['color'] == 'Crustose coralline algae', 'color'] = color_palette[1]
    top_8.loc[top_8['color'] == 'Sand', 'color'] = color_palette[2]
    top_8.loc[top_8['color'] == 'Red', 'color'] = color_palette[3]
    top_8.loc[top_8['color'] == 'Other fucoids', 'color'] = color_palette[4]
    top_8.loc[top_8['color'] == 'Coarse sand (with shell fragments)', 'color'] = color_palette[5]
    top_8.loc[top_8['color'] == 'Mud / silt (<64um)', 'color'] = color_palette[6]
    top_8.loc[top_8['color'] == 'Sand / mud (<2mm)', 'color'] = color_palette[7]
    top_8.loc[top_8['color'] == 'Turfing algae', 'color'] = color_palette[8]
    top_8.loc[top_8['color'] == 'Biota', 'color'] = color_palette[9]
    top_8.loc[top_8['color'] == 'Phyllospora comosa', 'color'] = color_palette[8]
    top_8.loc[top_8['color'] == 'Phyllospora', 'color'] = color_palette[11]

    subtitles = ['Label Distribution 1:1 and 1:10 Ratio', 'Label Distribution 1:10 Ratio', 'Label Distribution 1:1 Ratio Neighbour']
    titles = [f'123235 & 680037 Annotation Entries',f'680037 Annotation Entries', f'164161 Annotation Entries']
for idx, dfs in enumerate([df1_top8, df2_top8, df3_top8]):
    dfs = dfs.sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # creating the data values for the vertical y and horisontal x axis
    y = [1,2,3,4,5,6,7,8]


    matplotlib.rcParams['font.family'] = ['Source Han Sans TW', 'sans-serif']
    hfont = {'fontname':'sans-serif'}
    # using the pyplot.barh funtion for the horizontal bar
    plt.barh(y,dfs['percent'], color=dfs.color, label=dfs['label'], height=0.6)

    # Format the plot
    ax.set_xlabel('Percentage',fontsize=16)
    ax.set_ylabel('Top 8 Annotations',fontsize=16)
    #ax.set_title(f'Top 10 Labels - Dataset 1')
    plt.suptitle(subtitles[idx], fontsize=20, y=1.00, **hfont)
    plt.title(titles[idx], y=1.03, fontsize=16)
    plt.legend(fontsize=16)
    ax.set_ylim(ymin=0)
    plt.yticks(range(1,9), fontsize=16)
    plt.xticks(range(0,40,5), fontsize=16)
    ax.set_ylim([0.5, 8.5])

    plt.savefig('/home/leo/Documents/IMAS/Code/PyTorch/Annotation_Sets/Paper_Data/Figures/' + subtitles[idx]+'.pdf', format='pdf', bbox_inches='tight')
    # to show our graph
    plt.show()