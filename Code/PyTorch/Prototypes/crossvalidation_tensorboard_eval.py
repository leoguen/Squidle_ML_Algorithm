from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd

filepath = 'Ecklonia.csv'
perc = 18
method = 'f1_score'
method = 'accuracy'
f1_scores = []
valid = []


### Get all subfolders (percentages) in log ###
#directory = f'/pvol/logs/Macroalgae_Cross_Validation/perc_18/lightning_logs/'
#directory = f'/pvol/logs/Hardcoral_Cross_Validation/perc_20/lightning_logs/'
#directory = f'/pvol/logs/Seagrass_crossvalidation/perc_10/lightning_logs/'
directory = f'/pvol/logs/Eck_cross_validation_1/perc_18/lightning_logs/'   

version_list = next(os.walk(directory))[1]#[x[0] for x in os.walk(directory)]
for idx, entry in enumerate(version_list):
    version_list[idx] = int(re.sub("[^0-9]", "", entry))
version_list = sorted(version_list, reverse=False)
print(version_list)
### End Subfolders ###

if filepath == 'Ecklonia.csv':
    list = [0, 0, 0, 0, 0, 0, 0, 0]
else:
    list = [0, 0, 0]
test = [list.copy() for _ in range(len(version_list))]
test_avg = [0] * len(version_list)
path = ['valid', 'test']

for method in ['f1_score', 'accuracy']:
    for version in version_list:
        #log_dir = f"/media/leo/NESP_1/logs/Final_Grid_50/{run_name}/perc_{perc}/lightning_logs/version_0/{method}/valid/"
        log_dir = directory + f"version_{version}/{method}/valid/"
        for subdir, dirs, files in os.walk(log_dir):
            files.sort()
            #!!! SORT FILES BY NAME 
            file = files[-1]
            #file = "events.out.tfevents.1685810578.large-gpu-3"
            if file.startswith("events.out"):
                event_acc = EventAccumulator(os.path.join(subdir, file))
                event_acc.Reload()
                for tag in event_acc.Tags()["scalars"]:
                    if method in tag:
                        f1_score = event_acc.Scalars(tag)[-1].value
                        print(f1_score)
                        valid.append(f1_score)
    if method == 'f1_score':
        f1_valid = valid
        print(f1_valid)
        valid = []
    else:
        acc_valid = valid
        print(acc_valid)
        valid = []
    

    for idx, version in enumerate(version_list):
        log_dir = directory + f"version_{version}/test_{method}/test/"
        for subdir, dirs, files in os.walk(log_dir):
            for index, file in enumerate(files):
                if file.startswith("events.out"):
                    event_acc = EventAccumulator(os.path.join(subdir, file))
                    event_acc.Reload()
                    for tag in event_acc.Tags()["scalars"]:
                        if method in tag:
                            f1_score = event_acc.Scalars(tag)[-1].value
                            test[idx][index]=f1_score
        test_avg[idx] = sum(test[idx])/len(test[idx])
    if method == 'f1_score':
        f1_test = test
        f1_test_avg = test_avg 
        #print(f1_valid)
        test = [list.copy() for _ in range(len(version_list))]
        test_avg = [0] * len(version_list)
    else:
        acc_test = test
        acc_test_avg = test_avg 
        #print(f1_valid)
        test = [list.copy() for _ in range(len(version_list))]
        test_avg = [0] * len(version_list)

    #print(f1_test_avg)
#print(f1_test)

df = pd.DataFrame(zip(f1_valid, acc_valid, f1_test, f1_test_avg, acc_test, acc_test_avg), index =['1', '2', '3', '4', '5'], columns =['f1_valid', 'acc_valid', 'f1_test', 'f1_test_avg', 'acc_test', 'acc_test_avg' ])
print(df)
df.to_csv(filepath) 
