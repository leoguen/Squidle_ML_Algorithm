from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import numpy as np
import pandas as pd

plot_name = 'Crossvalidation_Results'
plot_path = '/home/leo/Documents/IMAS/Code/PyTorch/Annotation_Sets/Paper_Data/Figures/'
no_versions = [0,1,2]
res_df = pd.DataFrame()

for version in no_versions:
    og_dir = f"/media/leo/NESP_1/logs/Presentation_cross_validation/perc_18/lightning_logs/version_{version}/"
    row_data = {}
    
    for log_dir, dirs, files in os.walk(og_dir):
        for dir in dirs:
            if dir.startswith('test'): 
                if dir.startswith('test_probability'):
                    break
                method = dir
                for log_dir, subdir, files in os.walk(og_dir+dir):
                    #subdir = 'test'
                    for files in os.walk(os.path.join(og_dir,method, 'test')):
                        files = files[2]
                        files.sort()
                        for idx, file in enumerate(files):
                            if file.startswith("events.out"):
                                event_acc = EventAccumulator(os.path.join(og_dir,method, 'test', file))
                                event_acc.Reload()
                                for tag in event_acc.Tags()["scalars"]:
                                    column_name = f"{tag}_{idx}"
                                    row_data[column_name] = event_acc.Scalars(tag)[-1].value
                                    #Add Average row
                                    if idx == len(files)-1:
                                        column_name = f"{tag}_avg"
                                        row_data[column_name] = 0
                                        for idx in range(len(files)):
                                            row_data[column_name] += row_data[f"{tag}_{idx}"]
                                        row_data[column_name] = row_data[column_name]/len(files)
            else:
                method = dir
                for log_dir, subdirs, files in os.walk(og_dir+dir):
                    for subdir in subdirs:
                        train_or_valid = subdir
                        for files in os.walk(os.path.join(og_dir,method, train_or_valid)):
                            files = files[2]
                            files.sort()
                            file = files[-1]
                            if file.startswith("events.out"):
                                event_acc = EventAccumulator(os.path.join(og_dir,method ,train_or_valid, file))
                                event_acc.Reload()
                                for tag in event_acc.Tags()["scalars"]:
                                    column_name = f"{tag}_{train_or_valid}"
                                    row_data[column_name] = event_acc.Scalars(tag)[-1].value
    
    if version == 0:
        res_df = pd.DataFrame(columns=row_data.keys())
    
    res_df = res_df.append(row_data, ignore_index=True)
    
res_df.index = ["version_" + str(i) for i in no_versions]
res_df = res_df.sort_values(by='accuracy_valid', ascending=False)

res_df.to_csv(plot_name+'.csv')

bib_df = res_df[['accuracy_valid','accuracy_train','f1_score_valid','f1_score_train', 'test_accuracy_avg', 'test_f1_score_avg']].copy()
#bib_df.index = ["version_" + str(i) for i in range(no_versions)]

print(bib_df.to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format,
))  