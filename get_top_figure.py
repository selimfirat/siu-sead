import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

df = pd.read_csv("figures/get_top_parameter.csv")
df["Veri Kümesi"] = df["data"]

ax = sns.lineplot(x="get_top", y="ap", data=df, hue="Veri Kümesi", dashes=True, style="Veri Kümesi")

ax.set(xlabel='$\lambda$ Parametresi', ylabel='Ortalama Kesinlik', xlim=(0,1.01), ylim=(0,1.01))

# ax.get_legend().set_title("Veri Kümesi")

fig = ax.get_figure()

fig.savefig("figures/get_top_parameter.pdf")

plt.show()
