import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
file_path = '/jet/home/rbandaru/ravi/headneck/metadata/RAW_HNSCC_METADATA.csv'
df = pd.read_csv(file_path)
visit_column = df['Type of Visit']
visit_counts = visit_column.value_counts().to_dict()
visit_counts_df = pd.DataFrame(list(visit_counts.items()), columns=['Type of Visit', 'Count'])
plt.figure(figsize=(4, 3.5))
ax = sns.barplot(x='Type of Visit', y='Count', data=visit_counts_df, palette='coolwarm', width=0.5, hue=None)
plt.xlabel('Type of Visit', fontsize=14)
plt.ylabel('Count', fontsize=14)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='baseline', fontsize=12, color='black', 
                xytext=(0, 5), textcoords='offset points')
sns.despine()
plt.tight_layout()
plt.savefig('three_tp_dist.png', dpi=1000, bbox_inches='tight')
