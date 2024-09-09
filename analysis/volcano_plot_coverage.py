import glob
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# Load and merge coverage data
dfs = []
file_pattern = '/jet/home/rbandaru/ravi/headneck/processed_data/*.coverage.500kb_GCadjusted.bed'

for file in glob.glob(file_pattern):
    # Extract file name without extension for column name
    col_name = file.split('/')[-1].replace('.coverage.500kb_GCadjusted.bed', '')
    
    # Load the relevant columns into a DataFrame
    df = pd.read_csv(file, delimiter='\t', usecols=[0, 1, 2, 4], header=None, names=['chr', 'start', 'end', 'coverage'])
    
    # Rename the 'coverage' column
    df.columns = ['chr', 'start', 'end', col_name]
    
    # Append to list
    dfs.append(df)

# Merge all DataFrames on 'chr', 'start', and 'end' columns
final_df = pd.merge(dfs[0], dfs[1], on=['chr', 'start', 'end'], how='outer')  # Merge the first two DataFrames

for df in dfs[2:]:
    final_df = pd.merge(final_df, df, on=['chr', 'start', 'end'], how='outer')  # Merge the rest

datapoints = final_df.copy()

final_df = final_df.iloc[:, 3:]
final_df = final_df.dropna()

def calculate_column_averages(df_screen, final_df):
    # Extract IDs for responders and non-responders
    responders_ids = df_screen[df_screen['Treatment Response'] == 'Responder']['ID']
    non_responders_ids = df_screen[df_screen['Treatment Response'] == 'Non-Responder']['ID']
    
    # Transpose final_df to have IDs as rows
    final_df_transposed = final_df.T
    
    # Ensure that the transposed DataFrame has the IDs we are interested in
    if not set(responders_ids).issubset(final_df_transposed.index) or not set(non_responders_ids).issubset(final_df_transposed.index):
        raise ValueError("Some IDs from df_screen are not present in the columns of final_df.")
    
    # Filter the transposed DataFrame based on these IDs
    responders_df = final_df_transposed.loc[responders_ids]
    non_responders_df = final_df_transposed.loc[non_responders_ids]
    
    # Compute average for each column for responders and non-responders
    responder_avg = responders_df.mean(axis=0)
    non_responder_avg = non_responders_df.mean(axis=0)
    
    return responders_df, non_responders_df

# Path to the CSV file
csv_file_path = '/jet/home/rbandaru/ravi/headneck/metadata/RAW_HNSCC_METADATA.csv'

# Load the CSV file into a DataFrame
metadata_df = pd.read_csv(csv_file_path)

# Select specific columns
metadata_df = metadata_df[["ID", "Treatment Response", "Type of Visit"]]

# Drop rows with NaN values
metadata_df = metadata_df.dropna()

# Replace hyphens with underscores in the "ID" column
metadata_df["ID"] = metadata_df["ID"].str.replace('-', '_', regex=False)

# List of IDs to remove
ids_to_remove = ["Pilot_214", "Pilot_197", "Pilot_229", "Pilot_265", "Pilot2_47", "Pilot2_77", "Pilot_227"]

# Remove rows with the specified IDs
metadata_df = metadata_df[~metadata_df["ID"].isin(ids_to_remove)]

# Get subsets based on "Type of Visit"
df_screen = metadata_df[metadata_df["Type of Visit"] == "Screen"]
df_dayzero = metadata_df[metadata_df["Type of Visit"] == "Day 0"]
df_adjwk1 = metadata_df[metadata_df["Type of Visit"] == "Adj Wk 1"]

# Calculate column averages
responder_avg_screen, non_responder_avg_screen = calculate_column_averages(df_screen, final_df)
responder_avg_dayzero, non_responder_avg_dayzero = calculate_column_averages(df_dayzero, final_df)
responder_avg_adjwk1, non_responder_avg_adjwk1 = calculate_column_averages(df_adjwk1, final_df)

def create_volcano_plot(responder_avg_array, non_responder_avg_array, ax, title, threshold=0.05):
    # Compute fold changes
    fold_changes = np.mean(non_responder_avg_array, axis=0) / np.mean(responder_avg_array, axis=0)

    # Compute p-values using t-test
    p_values = np.array([
        ttest_ind(responder_avg_array[:, i], non_responder_avg_array[:, i]).pvalue
        for i in range(responder_avg_array.shape[1])
    ])

    volcano_df = pd.DataFrame({
        'FoldChange': fold_changes,
        'PValue': p_values
    })

    volcano_df['NegLog10PValue'] = -np.log10(volcano_df['PValue'])

    # Define a color mapping based on the threshold
    threshold_log = -np.log10(threshold)
    volcano_df['Color'] = np.where(volcano_df['NegLog10PValue'] > threshold_log, 'red', 'black')

    sns.scatterplot(data=volcano_df, x='FoldChange', y='NegLog10PValue', hue='Color', palette={'red': 'red', 'black': 'black'},
                    edgecolor=None, s=2, ax=ax)

    ax.axhline(threshold_log, color='red', linestyle='--')  # Significance threshold
    ax.axvline(x=1, color='gray', linestyle='--')  # Fold change threshold
    ax.set_xlabel('Fold Change')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title(title)
    ax.set_xlim(0.9, 1.1)  # Adjust x-axis limits as needed
    ax.legend().remove()

    # Return significant indices
    significant_indices = volcano_df.index[volcano_df['NegLog10PValue'] > threshold_log].tolist()
    return significant_indices

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(8, 6), sharey=True)

# Create volcano plots and get significant indices
significant_indices_screen = create_volcano_plot(responder_avg_screen.to_numpy(), non_responder_avg_screen.to_numpy(), axes[0], 'Screen')
significant_indices_dayzero = create_volcano_plot(responder_avg_dayzero.to_numpy(), non_responder_avg_dayzero.to_numpy(), axes[1], 'Day 0')
significant_indices_adjwk1 = create_volcano_plot(responder_avg_adjwk1.to_numpy(), non_responder_avg_adjwk1.to_numpy(), axes[2], 'Adj Wk 1')
filtered_data = datapoints.iloc[significant_indices_screen, :3]
filtered_data.to_csv('screen.csv', index=False, header=True, sep='\t')
filtered_data = datapoints.iloc[significant_indices_dayzero, :3]
filtered_data.to_csv('dayzero.csv', index=False, header=True, sep='\t')
filtered_data = datapoints.iloc[significant_indices_adjwk1, :3]
filtered_data.to_csv('adjwk1.csv', index=False, header=True, sep='\t')
plt.tight_layout()
plt.savefig('volcano_plot_coverage.png', dpi=1000, bbox_inches='tight')
