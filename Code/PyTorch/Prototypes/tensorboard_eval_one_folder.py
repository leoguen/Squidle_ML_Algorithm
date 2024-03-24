from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import matplotlib.pyplot as plt
import numpy as np
import re

add_highres = True
one_folder = True
method = 'f1_score'
#method = 'accuracy'
y_label = 'F1 Score'
#y_label = 'Accuracy'
x_label = 'Percentage Crop'
f1_scores = []
f1_valid = []

### Get all subfolders (percentages) in log ###
plot_name = 'Seagrass_2' + '_Neighbour_' + method
directory = f'/media/leo/NESP_1/logs/Seagrass_grid_perc_2/'
perc_list = next(os.walk(directory))[1]#[x[0] for x in os.walk(directory)]
for idx, entry in enumerate(perc_list):
    perc_list[idx] = int(re.sub("[^0-9]", "", entry))
perc_list = sorted(perc_list, reverse=False)
print(perc_list)
### End Subfolders ###

list = [0, 0, 0]
f1_test = [list.copy() for _ in range(len(perc_list))]
f1_test_avg = [0] * len(perc_list)
path = ['valid', 'test']
plot_path = '/home/leo/Documents/IMAS/Paper/Figures/'

### plot definitions ###
plt.figure(figsize=(8,3))
y_min = 0.5 #smallest value for y axis
legend_loc = 'lower right'
### plot definitions end ###
if one_folder:
    for perc in perc_list:
        #log_dir = f"/media/leo/NESP_1/logs/Final_Grid_50/{run_name}/perc_{perc}/lightning_logs/version_0/{method}/valid/"
        log_dir = directory + f"perc_{perc}/lightning_logs/version_0/{method}/valid/"
        for subdir, dirs, files in os.walk(log_dir):
            file = files[-1]
            if file.startswith("events.out"):
                event_acc = EventAccumulator(os.path.join(subdir, file))
                event_acc.Reload()
                for tag in event_acc.Tags()["scalars"]:
                    if method in tag:
                        f1_score = event_acc.Scalars(tag)[-1].value
                        f1_valid.append(f1_score)


    for idx, perc in enumerate(perc_list):
        log_dir = directory + f"perc_{perc}/lightning_logs/version_0/test_{method}/test/"
        for subdir, dirs, files in os.walk(log_dir):
            for index, file in enumerate(files):
                if file.startswith("events.out"):
                    event_acc = EventAccumulator(os.path.join(subdir, file))
                    event_acc.Reload()
                    for tag in event_acc.Tags()["scalars"]:
                        if method in tag:
                            f1_score = event_acc.Scalars(tag)[-1].value
                            f1_test[idx][index]=f1_score
        f1_test_avg[idx] = sum(f1_test[idx])/len(f1_test[idx])

    res_f1_valid = [None]*100 # new list with 99 entries
    res_f1_test_avg = [None]*100 # new list with 99 entries

    for list_idx, perc_idx in enumerate(perc_list):
        res_f1_valid[perc_idx] = f1_valid[list_idx]
        res_f1_test_avg[perc_idx] = f1_test_avg[list_idx]

    x_values = [i for i in range(0, 100, 1)]
    x_values[0] = 1
    f1_valid_series = np.array(res_f1_valid).astype(np.double)
    f1_valid_mask = np.isfinite(f1_valid_series)
    f1_test_series = np.array(res_f1_test_avg).astype(np.double)
    f1_test_mask = np.isfinite(f1_test_series)
    xs = np.arange(100)
    # Create line plots
    plt.plot(xs[f1_valid_mask], f1_valid_series[f1_valid_mask], label='Validation Results', marker='o', markersize=6)
    plt.plot(xs[f1_test_mask], f1_test_series[f1_test_mask], label='Test Results', marker='s', markersize=6)
    #plt.plot(x_values, f1_test_avg, label='High Resolution Validation Results', marker='s', markersize=6)
    #plt.plot(x_values, f1_test_avg, label='High Resolution Test Results', marker='s', markersize=6)

    # Add labels and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=legend_loc)

    # Highlight maximum f1_valid value
    max_no_nan = [0 if v is None else v for v in res_f1_valid]
    max_f1_valid = max(max_no_nan)

    min_f1_test_avg = min(f1_test_avg)
    max_index = res_f1_valid.index(max_f1_valid)
    #max_index = max_index -1
    plt.plot(x_values[max_index], max_f1_valid, marker='o', markersize=10, color='red')
    plt.text(x_values[max_index], max_f1_valid - 0.05,' '+str(x_values[max_index])+'%', color= 'red', fontsize=10, fontweight='bold')
    plt.plot([x_values[max_index], x_values[max_index]], [0, max_f1_valid], linestyle='--', color='red')

    # Adjust x-axis ticks and limits
    xticks = [i for i in range(0, 100, 5)]
    xticks[0] = 1
    plt.xticks(xticks)
    plt.xlim(0, 100)
    plt.ylim(y_min ,1)

    plt.savefig(plot_path+plot_name+'_high_res.pdf', format='pdf', bbox_inches='tight')
    # Show plot
    plt.show()
