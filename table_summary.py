import pandas as pd
import numpy as np
import csv

def count_instances(csv_file):
    df = pd.read_csv(csv_file)
    timeout_df = df[df['TimeDominance'] == 3600]
    return len(df), len(timeout_df)

def shifted_geometric_mean(series, shift=1):
    # Remove non-numeric values and NaNs
    series = pd.to_numeric(series, errors='coerce').dropna()
    shifted_series = series + shift
    log_sum = np.sum(np.log(shifted_series))
    return np.exp(log_sum / len(shifted_series)) - shift

def calculate_shifted_geometric_mean(csv_file, selected_columns, shift=1):
    df = pd.read_csv(csv_file)
    df = df[df['TimeDominance'] != 3600]
    result = {}
    for column in selected_columns:
        if column in df.columns:
            result[column] = shifted_geometric_mean(df[column], shift)
        else:
            print(f"Column {column} not found in the CSV file.")
    return result

def calculate_mean(csv_file, selected_columns):
    df = pd.read_csv(csv_file)
    df = df[df['TimeDominance'] != 3600]
    result = {}
    for column in selected_columns:
        if column in df.columns:
            result[column] = df[column].mean()
        else:
            print(f"Column {column} not found in the CSV file.")
    return result

csv_files = {
    # 'setcover': 'results/setcover/valid_500r_1000c_0.05d/gurobi_0_robust_1_df_0_ratio_0.6_0.0_plow_0.9999_pup_0.0_gap_0.001_heuristics_0.05.csv',
    'setcover': 'results/setcover/valid_500r_1000c_0.05d/gurobi_0_robust_1_df_0_ratio_0.4_0.0_plow_0.9_pup_0.0_gap_0.001_heuristics_0.05.csv',
    'cauctions': 'results/cauctions/valid_200_1000/gurobi_0_robust_1_df_0_ratio_0.6_0.0_plow_0.999999999_pup_0.0_gap_0.001_heuristics_0.05.csv',
}

datasets = csv_files.keys()
shift = 0
output_file = f'results/branching.csv'
# selected_columns = ['ori_time','ori_warm_time','mvb_time','ori_best_time','ori_warm_best_time','TimeDominance']
selected_columns = ['ori_time','ori_warm_time','mvb_time_all']

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Dataset']+ ['instances', 'timeout_instances'] + selected_columns)

    for dataset in datasets:
        csv_file = csv_files[dataset]    
        instances, timeout_instances = count_instances(csv_file)    
        result = calculate_shifted_geometric_mean(csv_file, selected_columns, shift)
        result_mean = calculate_mean(csv_file, selected_columns)
        row = [dataset] + [instances, timeout_instances] + [result.get(column, 'N/A') for column in selected_columns] + [result_mean.get(column, 'N/A') for column in selected_columns]
        writer.writerow(row)
        print(f"{row}")

print(f"Results have been written to {output_file}")
