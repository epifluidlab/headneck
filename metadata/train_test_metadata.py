import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


file_path = '/jet/home/rbandaru/ravi/headneck/metadata/RAW_HNSCC_METADATA.csv'
df = pd.read_csv(file_path)
df = df.dropna(subset=['Treatment Response'])
train_df = df[df['Institute'].isin([1, 2, 3, 4])]
test_df = df[df['Institute'].isin([5, 6])]
num_responder_train = train_df['Treatment Response'].value_counts().get("Responder", 0)
num_nonresponder_train = train_df['Treatment Response'].value_counts().get("Non-Responder", 0)
num_responder_test = test_df['Treatment Response'].value_counts().get("Responder", 0)
num_nonresponder_test = test_df['Treatment Response'].value_counts().get("Non-Responder", 0)
data = {
    'Responder': [num_responder_train, num_responder_test ],
    'Non-Responder': [num_nonresponder_train, num_nonresponder_test]
}
df_counts = pd.DataFrame(data, index=['Responder', 'Non-Responder'])
ax = df_counts.plot(kind='bar', stacked=True, figsize=(4, 6), color=['#377eb8', '#e41a1c'])
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_xticklabels(['Train', 'Test'], rotation=0, ha='center', fontsize=10)  # Update x-axis labels
ax.legend(title='Treatment Response', title_fontsize=10, fontsize=10, frameon=False, labels=['Responder', 'Non-Responder'])  # Update legend labels
plt.savefig('train_test_metadata.png', dpi=1000, bbox_inches='tight')
plt.show()
