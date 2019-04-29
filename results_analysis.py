import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, glob

data_dir = os.path.join('results','mask_lh_m1s1_new')

all_result_files = glob.glob(data_dir + "/*.csv")

dfs = []
for filename in all_result_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    dfs.append(df)

results = pd.concat(dfs, axis=0, ignore_index=True)

ax=sns.barplot(x='detrend', y='clf_acc', data=results)
plt.title('SG filtering: 240s frame length')
plt.ylim([.25,1])
plt.ylabel('decoder accuracy')
plt.xlabel('filter order')
plt.show()
