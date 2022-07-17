import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

def calc_pval(input):
    if input >= 0.05:
        return 'p >= 0.05'
    elif input < 0.05 and input > 0.01:
        return 'p < 0.05'
    elif input < 0.01 and input > 0.001:
        return 'p < 0.01'
    elif input < 0.001:
        return 'p < 0.001'

df = pd.read_excel('2022 O2_gesamt_Stud.xlsx')

df_sorted = df[['Aufenthaltsdauer', 'Molarität', 'Kosten/Aufenthalt (J)']]

df_sorted_05 = df_sorted[df_sorted['Molarität'] == 0.5]
df_sorted_15 = df_sorted[df_sorted['Molarität'] == 1.5]

resultA1 = stats.linregress(df_sorted_05['Aufenthaltsdauer'], df_sorted_05['Kosten/Aufenthalt (J)'])
linearModelA1 = LinearRegression()
linearModelA1.fit(df_sorted_05['Aufenthaltsdauer'].to_numpy().reshape(-1, 1), df_sorted_05['Kosten/Aufenthalt (J)'])
regressionA1 = linearModelA1.predict(np.array([df_sorted_05['Aufenthaltsdauer'].min(), df_sorted_05['Aufenthaltsdauer'].max()]).reshape(-1, 1))

resultA2 = stats.linregress(df_sorted_15['Aufenthaltsdauer'], df_sorted_15['Kosten/Aufenthalt (J)'])
linearModelA2 = LinearRegression()
linearModelA2.fit(df_sorted_15['Aufenthaltsdauer'].to_numpy().reshape(-1, 1), df_sorted_15['Kosten/Aufenthalt (J)'])
regressionA2 = linearModelA2.predict(np.array([df_sorted_15['Aufenthaltsdauer'].min(), df_sorted_15['Aufenthaltsdauer'].max()]).reshape(-1, 1))

sns.set()
custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
sns.set_theme(style='ticks', rc=custom_params)

fig0, ax0 = plt.subplots(figsize=(30, 15))
ax0.set_xlim(20, 180)
ax0.set_ylim(0, 10)

plotA1 = ax0.scatter(df_sorted_05['Aufenthaltsdauer'], df_sorted_05['Kosten/Aufenthalt (J)'], label='Cost per stay [J] 0.5 Molar', color='red', marker="o", alpha=0.5)
plotA2 = ax0.scatter(df_sorted_15['Aufenthaltsdauer'], df_sorted_15['Kosten/Aufenthalt (J)'], label='Cost per stay [J] 1.5 Molar', color='blue', marker="o", alpha=0.5)
plotC3 = ax0.plot([df_sorted_05['Aufenthaltsdauer'].min(), df_sorted_05['Aufenthaltsdauer'].max()], regressionA1, color='k', linestyle='dashed', label='LinearRegression Cost per stay [J] 0.5 Molar')
plotC4 = ax0.plot([df_sorted_15['Aufenthaltsdauer'].min(), df_sorted_15['Aufenthaltsdauer'].max()], regressionA2, color='k', label='LinearRegression Cost per stay [J] 1.5 Molar')

StringA1 = 'Linear Regression' + f' 0.5 Mol\n$R^2$ = {round(resultA1.rvalue, 2)}, p-value = {calc_pval(resultA1.pvalue)}'
StringA2 = 'Linear Regression' + f' 1.5 Mol\n$R^2$ = {round(resultA2.rvalue, 2)}, p-value = {calc_pval(resultA2.pvalue)}'

ax0.annotate(StringA1, xy=(150, 4), fontsize=25)
ax0.annotate(StringA2, xy=(150, 5), fontsize=25)

ax0.tick_params(axis='both', which='major', labelsize=20)
ax0.legend(loc='best', fontsize=25)
ax0.set_ylabel('Cost per stay [J]', labelpad=10, fontsize=25)
ax0.set_xlabel('Duration of stay [s]', labelpad=10, fontsize=25)
fig0.tight_layout()
plt.savefig('costperstay_time.png')