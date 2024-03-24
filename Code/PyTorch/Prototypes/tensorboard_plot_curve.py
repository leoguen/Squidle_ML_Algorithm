import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Define the log files path
log_dir = "/media/leo/NESP_1/logs/Eck_cross_validation/perc_18/lightning_logs/version_3/"
plot_path = '/home/leo/Documents/IMAS/Paper/Figures/'

# Define the metrics to plot
metrics = ["accuracy", "f1_score"]

# Define the subfolders for train and validation
subfolders = ["train", "valid"]

# Define the color of the line after epoch 15
color_after_epoch = 50

# Loop through each metric and subfolder
for metric in metrics:
    for idx, subfolder in enumerate(subfolders):
        # Define the event file path
        event_file = os.path.join(log_dir, metric,subfolder)

        # Load the event file
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()

        # Get the scalar data for the metric
        scalar_data = event_acc.Scalars(metric)

        # Extract the epoch and metric value from the scalar data
        if idx == 0: 
            epoch_train = range(len(scalar_data)) #[scalar.step for scalar in scalar_data]
            value_train =[scalar.value for scalar in scalar_data]
            # Define the color of the line based on the epoch
            color_train = "blue"
            if color_after_epoch is not None:
                color_train = "orange"
                for i, e in enumerate(epoch_train):
                    if e > color_after_epoch:
                        color_train = "blue"
                        break
        else:
            epoch_valid = range(len(scalar_data)) #[scalar.step for scalar in scalar_data]
            value_valid = [scalar.value for scalar in scalar_data]
            # Define the color of the line based on the epoch
            color_valid = "orange"
            if color_after_epoch is not None:
                color_valid = "orange"
                for i, e in enumerate(epoch_valid):
                    if e > color_after_epoch:
                        color_valid = "red"
                        break
        
    # split the data into two parts based on x=15
    epoch_train1, value_train1 = epoch_train[:color_after_epoch+1], value_train[:color_after_epoch+1]  # data before x=15
    epoch_train2, value_train2 = epoch_train[color_after_epoch:], value_train[color_after_epoch:]  # data after x=15

    # split the data into two parts based on x=15
    epoch_valid1, value_valid1 = epoch_valid[:color_after_epoch+1], value_valid[:color_after_epoch+1]  # data before x=15
    epoch_valid2, value_valid2 = epoch_valid[color_after_epoch:], value_valid[color_after_epoch:]  # data after x=15
    
    #Define plot size
    plt.figure(figsize=(8,3))
    
    # plot the first part and second train part with different color
    plt.plot(epoch_train1, value_train1, color='tab:orange', label='Train Results')
    plt.plot(epoch_train2, value_train2, color='#F5B375')
    
    #plot final value as text
    plt.text(epoch_train[len(epoch_train)-1], value_train[len(value_train)-1]-0.006," {:.3f}".format(value_train[len(value_train)-1]), color= 'black', fontsize=10)


    # plot the first part and second value part with different color
    plt.plot(epoch_valid1, value_valid1, color='tab:blue', label='Validation Results')
    plt.plot(epoch_valid2, value_valid2, color='#8FCEFF')
    # plot final value as text
    plt.text(epoch_valid[len(epoch_valid)-1], value_valid[len(value_valid)-1]-0.006," {:.3f}".format(value_valid[len(value_valid)-1]), color= 'black', fontsize=10)
    # plot legend
    plt.legend(loc='lower right')
    
    # Define plot design
    plt.xlim(0, 55)
    min_y = 0.8
    max_y = 1.05
    #plt.yticks(range(min_y*100, 100, 0.05*100)/100)
    plt.ylim(0.85 ,1.0)

    # Add title and axis labels
    #plt.title(f"{metric.capitalize()}")
    plt.xlabel("Epoch")
    if metric == 'f1_score':
        ylabel = 'F1 Score'
    else: ylabel = metric.capitalize()
    plt.ylabel(ylabel)

    plt.savefig(plot_path+'Ecklonia_'+metric+'_crossvalidation.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(plot_path+'Ecklonia_'+metric+'_crossvalidation.png', format='png', bbox_inches='tight')
    # Show the plot
    plt.show()
