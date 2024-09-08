import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

fragment_counts = pd.read_csv('/jet/home/rbandaru/ravi/headneck/high_quality_fragment_counts.txt', sep='\t', header=None)
metadata = pd.read_csv('/jet/home/rbandaru/ravi/headneck/metadata/RAW_HNSCC_METADATA.csv')
metadata['ID'] = metadata['ID'].str.replace('-', '_', regex=False)
merged_df = pd.merge(fragment_counts, metadata, left_on=fragment_counts.columns[0], right_on='ID')
merged_df = merged_df.drop(columns=fragment_counts.columns[0])
merged_df = merged_df.rename(columns={fragment_counts.columns[1]: 'Fragment Count'})
merged_df = merged_df[["Fragment Count", "ID", "Institute"]]
merged_df['Fragment Count'] = merged_df['Fragment Count'] / 1e6

def boxplot_stats_log(data):
    data_log = np.log10(data[data > 0])
    Q1_log = data_log.quantile(0.25)
    Q3_log = data_log.quantile(0.75)
    IQR_log = Q3_log - Q1_log
    lower_whisker_log = max(data_log.min(), Q1_log - 1.5 * IQR_log)
    upper_whisker_log = min(data_log.max(), Q3_log + 1.5 * IQR_log)
    return Q1_log, Q3_log, lower_whisker_log, upper_whisker_log, data_log.median()

plt.figure(figsize=(5, 6))

institutes = merged_df['Institute'].unique()

for i, institute in enumerate(institutes):
    subset = merged_df[merged_df['Institute'] == institute]
    data = subset['Fragment Count']
    ids = subset['ID']

    Q1_log, Q3_log, lower_whisker_log, upper_whisker_log, median_log = boxplot_stats_log(data)
    Q1 = 10**Q1_log
    Q3 = 10**Q3_log
    lower_whisker = 10**lower_whisker_log
    upper_whisker = 10**upper_whisker_log
    median = 10**median_log

    plt.plot([i, i], [lower_whisker, Q1], color='black')
    plt.plot([i, i], [Q3, upper_whisker], color='black')
    plt.plot([i - 0.2, i + 0.2], [Q1, Q1], color='black')
    plt.plot([i - 0.2, i + 0.2], [Q3, Q3], color='black')
    plt.fill_between([i - 0.2, i + 0.2], Q1, Q3, color='lightblue', alpha=0.5)
    plt.plot([i - 0.2, i + 0.2], [median, median], color='red')

    jitter = np.random.uniform(-0.1, 0.1, size=len(data))
    plt.scatter(i + jitter, data, s=10, color='black', alpha=0.6)

    for x, y, label in zip(i + jitter, data, ids):
        if y + 1 < lower_whisker and label != "Pilot2_77":
            plt.text(x + 0.5, y, label, fontsize=8, ha='center', va='bottom')
        elif y + 1 < lower_whisker:
            plt.text(x - 0.4, y, label, fontsize=8, ha='center', va='bottom')

plt.yscale('log')
formatter = ticker.FuncFormatter(lambda x, _: f'{int(x):,}M')
plt.gca().yaxis.set_major_formatter(formatter)
plt.xlabel('Institute')
plt.ylabel('cfDNA Fragment Count')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('coverage_distribution.png', dpi=1000, bbox_inches='tight')
