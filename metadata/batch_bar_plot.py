import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

file_path = '/jet/home/rbandaru/ravi/headneck/metadata/RAW_HNSCC_METADATA.csv'
df = pd.read_csv(file_path)[["Institute", "Treatment Response"]].fillna("Missing")
df_counts = df.groupby(["Institute", "Treatment Response"]).size().unstack(fill_value=0)
order = ["Responder", "Non-Responder", "Missing"]
df_counts = df_counts[order]
ax = df_counts.plot(kind='bar', stacked=True, figsize=(5, 5), color=['#add8e6', '#377eb8', '#e41a1c'])
ax.set_xlabel('Institute', fontsize=12, labelpad=30)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)
ax.legend(title='Treatment Response', title_fontsize=10, fontsize=10, frameon=False)
def add_bracket(ax, start, end, label):
    x_start = start - 0.5
    x_end = end + 0.5
    y_pos = -10  
    bracket_height = -10 
    ax.text((x_start + x_end) / 2, bracket_height - 0.1, label, ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='none', pad=2))
add_bracket(ax, 0, 3, 'Discovery\nCohort (1-4)')
add_bracket(ax, 4, 5, 'Validation\nCohort (5-6)')
plt.savefig('batch_bar_plot.png', dpi=1000, bbox_inches='tight')
plt.show()
