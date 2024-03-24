from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import matplotlib.pyplot as plt
import numpy as np

add_highres = True
method = 'f1_score'
y_label = 'F1 Score'
x_label = 'Percentage Crop'
run_name = 'Neighbour'
y_min = 0.5 #smallest value for y axis
f1_scores = []
f1_valid = []
list = [0, 0, 0, 0, 0, 0, 0, 0]
f1_test = [list.copy() for _ in range(20)]
f1_test_avg = [0] * 20
path = ['valid', 'test']
plot_name = run_name + '_Valid_Test_' + method
plot_path = '/home/leo/Documents/IMAS/Code/PyTorch/Annotation_Sets/Paper_Data/Figures/'
# Get Validation Results
for perc in range(0,99,5):
    if perc == 0: perc = 1
    log_dir = f"/media/leo/NESP_1/logs/Final_Grid_50/{run_name}/perc_{perc}/lightning_logs/version_0/{method}/valid/"
    for subdir, dirs, files in os.walk(log_dir):
        file = files[-1]
        if file.startswith("events.out"):
            event_acc = EventAccumulator(os.path.join(subdir, file))
            event_acc.Reload()
            for tag in event_acc.Tags()["scalars"]:
                if method in tag:
                    f1_score = event_acc.Scalars(tag)[-1].value
                    f1_valid.append(f1_score)

# Get Test Results
for idx, perc in enumerate(range(0,99,5)):
    if perc == 0: perc = 1
    log_dir = f"/media/leo/NESP_1/logs/Final_Grid_50/{run_name}/perc_{perc}/lightning_logs/version_0/test_{method}/test/"
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

    #f1_test_avg[idx] = sum(f1_test[idx][0:4])/len(f1_test[idx][0:4])

if add_highres:
    # Get Test Results
    f1_test_avg
    res_test_avg  = [0] * 7
    f1_res_scores = []
    f1_res_valid = []
    list = [0, 0, 0, 0, 0, 0, 0, 0]
    f1_res_test = [list.copy() for _ in range(7)]
        
        # Get Validation Results
    for perc in range(10,24,2):
        if perc == 0: perc = 1
        log_dir = f"/media/leo/NESP_1/logs/Final_Grid_50/{run_name}_high_res/perc_{perc}/lightning_logs/version_0/{method}/valid/"
        for subdir, dirs, files in os.walk(log_dir):
            file = files[-1]
            if file.startswith("events.out"):
                event_acc = EventAccumulator(os.path.join(subdir, file))
                event_acc.Reload()
                for tag in event_acc.Tags()["scalars"]:
                    if method in tag:
                        f1_score = event_acc.Scalars(tag)[-1].value
                        f1_res_valid.append(f1_score)
    
    # Get Test results
    for idx, perc in enumerate(range(10,24,2)):
        if perc == 0: perc = 1
        log_dir = f"/media/leo/NESP_1/logs/Final_Grid_50/{run_name}_high_res/perc_{perc}/lightning_logs/version_0/test_{method}/test/"
        for subdir, dirs, files in os.walk(log_dir):
            for index, file in enumerate(files):
                if file.startswith("events.out"):
                    event_acc = EventAccumulator(os.path.join(subdir, file))
                    event_acc.Reload()
                    for tag in event_acc.Tags()["scalars"]:
                        if method in tag:
                            f1_res_score = event_acc.Scalars(tag)[-1].value
                            f1_res_test[idx][index]=f1_res_score
        res_test_avg[idx] = sum(f1_res_test[idx])/len(f1_res_test[idx])

    res_f1_valid = [None]*100 # new list with 99 entries
    res_f1_test_avg = [None]*100 # new list with 99 entries



    #Adjust old values to 100 entries
    new_f1_valid = [] # new list with 99 entries
    new_f1_test_avg = [] # new list with 99 entries
    for list, new_list in [[f1_valid,new_f1_valid], [f1_test_avg, new_f1_test_avg]]:
        for i in range(0, len(list)):
            new_list.append(list[i]) # add the current entry from the original list
            for j in range(0, 4):
                new_list.append(None) # add 4 times the value 'None'
            if len(new_list) == 99:
                break # stop when the new list has 99 entries
    #f1_valid
        # Create x-axis values
    for old_i,i in enumerate(range(10,24,2)):
        new_f1_test_avg[i] = res_test_avg[old_i]
        new_f1_valid[i] = f1_res_valid[old_i]

    x_values = [i for i in range(0, 100, 1)]
    x_values[0] = 1
    f1_valid_series = np.array(new_f1_valid).astype(np.double)
    f1_valid_mask = np.isfinite(f1_valid_series)
    f1_test_series = np.array(new_f1_test_avg).astype(np.double)
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
    plt.legend()

    # Highlight maximum f1_valid value
    max_no_nan = [0 if v is None else v for v in new_f1_valid]
    max_f1_valid = max(max_no_nan)

    min_f1_test_avg = min(f1_test_avg)
    max_index = new_f1_valid.index(max_f1_valid)
    #max_index = max_index -1
    plt.plot(x_values[max_index], max_f1_valid, marker='o', markersize=10, color='red')
    plt.text(x_values[max_index], max_f1_valid + 0.01,' '+str(x_values[max_index])+'%', color= 'red', fontsize=10, fontweight='bold')
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
else:
    # Create x-axis values
    x_values = [i for i in range(0, 100, 5)]
    x_values[0] = 1

    # Create line plots
    plt.plot(x_values, f1_valid, label='Validation Results', marker='o', markersize=6)
    plt.plot(x_values, f1_test_avg, label='Test Results', marker='s', markersize=6)

    # Add labels and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    # Highlight maximum f1_valid value
    max_f1_valid = max(f1_valid)
    min_f1_test_avg = min(f1_test_avg)
    max_index = f1_valid.index(max_f1_valid)
    plt.plot(x_values[max_index], max_f1_valid, marker='o', markersize=10, color='red')
    #plt.text(x_values[max_index], max_f1_valid,' '+str(x_values[max_index])+'%', color= 'black', fontsize=10, fontweight='bold')
    plt.plot([x_values[max_index], x_values[max_index]], [0, max_f1_valid], linestyle='--', color='red')

    # Adjust x-axis ticks and limits
    plt.xticks(x_values)
    plt.xlim(0, 100)
    plt.ylim(y_min ,1)

    plt.savefig(plot_path+plot_name+'.pdf', format='pdf', bbox_inches='tight')
    # Show plot
    plt.show()

