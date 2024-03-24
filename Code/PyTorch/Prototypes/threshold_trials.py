import pandas as pd
import numpy as np

# Import CSV file as pandas dataframe
df = pd.read_csv('/pvol/logs/Train_General_vs_CSV/General/299/lightning_logs/version_1/test_results_NSW_Broughton.csv', index_col=0)

# Set thresholds for prediction
thresholds = [x/100 for x in range(50, 100)]
columns = ['TP', 'TN', 'FP', 'FN']
results_df = pd.DataFrame(columns=columns)

# Iterate over thresholds
for threshold in thresholds:
    # Calculate column name for current threshold
    column_name = 'prediction_'+str(threshold)
    # Calculate prediction based on threshold
    df[column_name] = df.iloc[:, 1] > threshold
    df[column_name] = df[column_name].astype(int)
    # Determine whether it is a True Positive, True Negative, False Positive, or False Negative Prediction
    df['TP'] = ((df[column_name] == 1) & (df['Truth'] == True)).astype(int)
    df['TN'] = ((df[column_name] == 0) & (df['Truth'] == False)).astype(int)
    df['FP'] = ((df[column_name] == 1) & (df['Truth'] == False)).astype(int)
    df['FN'] = ((df[column_name] == 0) & (df['Truth'] == True)).astype(int)
    # Calculate sum of TP, TN, FP, and FN
    tp = df['TP'].sum()
    tn = df['TN'].sum()
    fp = df['FP'].sum()
    fn = df['FN'].sum()
    # Calculate precision, recall, and f1-score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    # Store results in a dictionary
    results_dict = {'Threshold': threshold, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'Precision': precision, 'Recall': recall, 'F1-Score': f1_score}
    # Append results to results dataframe
    results_df = results_df.append(results_dict, ignore_index=True)

# Save results to CSV file
results_df.to_csv('/pvol/logs/Train_General_vs_CSV/General/299/lightning_logs/version_1/Analyze_NSW_Broughton.csv', index=False)
print(results_df)
