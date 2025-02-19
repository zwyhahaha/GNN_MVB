import pandas as pd
import numpy as np
import csv

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
    'setcover': 'results/setcover/valid_500r_1000c_0.05d/gurobi_robust_0_plow_0.99_pup_0.0_gap_0.01_normalize_0_heuristics_0.05.csv',
    'cauctions': 'results/cauctions/valid_200_1000/gurobi_robust_0_ratio_0.6_plow_0.99_pup_0.999_gap_0.01_heuristics_1.0.csv',
    'indset': 'results/indset/valid_1000_4/gurobi_robust_0_plow_0.9_pup_0.9_gap_0.01_normalize_0_heuristics_0.05.csv',
    'fcmnf': 'results/fcmnf/valid/gurobi_1_robust_0_df_1_ratio_0.6_plow_0.99_pup_0.0_gap_0.01_heuristics_1.0.csv',
    'fcmnf2': 'results/fcmnf/valid/gurobi_0_robust_0_df_1_ratio_0.1_plow_0.6_pup_0.0_gap_0.01_heuristics_1.0.csv',
    'gisp': 'results/gisp/valid/gurobi_robust_0_plow_0.99999_pup_0.0_gap_0.01_normalize_0_heuristics_0.05.csv'
}

csv_files = {
    'indset': 'results/indset/valid_1000_4/gurobi_0_robust_0_df_0_tmvb_0.999_plow_0.99_pup_0.99_gap_0.01_heuristics_1.0.csv',
    'setcover': 'results/setcover/valid_500r_1000c_0.05d/gurobi_robust_0_plow_0.99_pup_0.0_gap_0.01_normalize_0_heuristics_0.05.csv',
    'setcover_df': 'results/setcover/valid_500r_1000c_0.05d/gurobi_0_robust_0_df_1_tmvb_0.9999_plow_0.999_pup_0.999_gap_0.01_heuristics_1.0.csv',
    'gisp': 'results/gisp/valid/gurobi_1_robust_0_df_1_ratio_0.4_plow_0.99_pup_0.0_gap_0.05_heuristics_1.0.csv',
    'cauctions': 'results/cauctions/valid_200_1000/gurobi_0_robust_0_df_1_tmvb_0.8_plow_0.9_pup_0.999_gap_0.001_heuristics_1.0.csv',
    'fcmnf': 'results/fcmnf/valid/gurobi_1_robust_0_df_1_ratio_0.8_plow_0.999_pup_0.99999_gap_0.001_heuristics_1.0.csv',
    'fcmnf2': 'results/fcmnf/valid/gurobi_0_robust_0_df_1_tmvb_0.9999_plow_0.9999_pup_0.99999999999_gap_0.001_heuristics_1.0.csv',
}

datasets = csv_files.keys()
shift = 0
output_file = f'results/SGM{shift}.csv'
selected_columns = ['ori_time','ori_warm_time','mvb_time','ori_best_time','ori_warm_best_time','TimeDominance']

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Dataset'] + selected_columns)

    for dataset in datasets:
        csv_file = csv_files[dataset]        
        result = calculate_shifted_geometric_mean(csv_file, selected_columns, shift)
        result_mean = calculate_mean(csv_file, selected_columns)
        row = [dataset] + [result.get(column, 'N/A') for column in selected_columns] + [result_mean.get(column, 'N/A') for column in selected_columns]
        writer.writerow(row)
        print(f"{row}")

print(f"Results have been written to {output_file}")
