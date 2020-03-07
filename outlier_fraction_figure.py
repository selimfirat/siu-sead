import matplotlib

import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
matplotlib.rc('font', family='Arial')

data = "satellite"

df = pd.read_csv(f"figures/outlier_fraction_{data}.csv")
df["Model"] = df["model"]

ax = sns.lineplot(x="fraction", y="ap", data=df, hue="Model", style="Model")

ax.set(xlabel='Anomali Oranı', ylabel='Ortalama Kesinlik', xlim=(0,1), ylim=(0,1.01))

fig = ax.get_figure()

ax.get_legend().texts[1].set_text("IO")
ax.get_legend().texts[2].set_text("k-EYK")
ax.get_legend().texts[3].set_text("Önerilen Model")
ax.get_legend().texts[4].set_text("TS-DVM")
fig.savefig(f"figures/outlier_fraction_{data}.pdf")

plt.show()