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

df = pd.read_excel('2022_Waage_gesamt_Stud.xlsx')
df_sorted = df[['Gew Ldg (mg)', 'TM (mg)', 'Ta (°C)', 'Gew Abflg (mg)', 'Molarität']]

df_sorted.dropna(inplace=True)

df_sorted_05 = df_sorted[df_sorted['Molarität'] == 0.5]
df_sorted_15 = df_sorted[df_sorted['Molarität'] == 1.5]

resultA1 = stats.linregress(df_sorted_05['Ta (°C)'], df_sorted_05['Gew Ldg (mg)'].astype(np.float64))
linearModelA1 = LinearRegression()
linearModelA1.fit(df_sorted_05['Ta (°C)'].to_numpy().reshape(-1, 1), df_sorted_05['Gew Ldg (mg)'])
regressionA1 = linearModelA1.predict(np.array([df_sorted_05['Ta (°C)'].max(), df_sorted_05['Ta (°C)'].min()]).reshape(-1, 1))

resultA2 = stats.linregress(df_sorted_15['Ta (°C)'], df_sorted_15['Gew Ldg (mg)'].astype(np.float64))
linearModelA2 = LinearRegression()
linearModelA2.fit(df_sorted_15['Ta (°C)'].to_numpy().reshape(-1, 1), df_sorted_15['Gew Ldg (mg)'])
regressionA2 = linearModelA2.predict(np.array([df_sorted_15['Ta (°C)'].max(), df_sorted_15['TM (mg)'].min()]).reshape(-1, 1))

resultB1 = stats.linregress(df_sorted_05['Ta (°C)'], df_sorted_05['TM (mg)'].astype(np.float64))
linearModelB1 = LinearRegression()
linearModelB1.fit(df_sorted_05['Ta (°C)'].to_numpy().reshape(-1, 1), df_sorted_05['TM (mg)'])
regressionB1 = linearModelB1.predict(np.array([df_sorted_05['Ta (°C)'].max(), df_sorted_05['Ta (°C)'].min()]).reshape(-1, 1))

resultB2 = stats.linregress(df_sorted_15['Ta (°C)'], df_sorted_15['TM (mg)'].astype(np.float64))
linearModelB2 = LinearRegression()
linearModelB2.fit(df_sorted_15['Ta (°C)'].to_numpy().reshape(-1, 1), df_sorted_15['TM (mg)'])
regressionB2 = linearModelB2.predict(np.array([df_sorted_15['Ta (°C)'].max(), df_sorted_15['TM (mg)'].min()]).reshape(-1, 1))

resultC1 = stats.linregress(df_sorted_05['Ta (°C)'], df_sorted_05['Gew Abflg (mg)'].astype(np.float64))
linearModelC1 = LinearRegression()
linearModelC1.fit(df_sorted_05['Ta (°C)'].to_numpy().reshape(-1, 1), df_sorted_05['Gew Abflg (mg)'])
regressionC1 = linearModelC1.predict(np.array([df_sorted_05['Ta (°C)'].max(), df_sorted_05['Ta (°C)'].min()]).reshape(-1, 1))

resultC2 = stats.linregress(df_sorted_15['Ta (°C)'], df_sorted_15['Gew Abflg (mg)'].astype(np.float64))
linearModelC2 = LinearRegression()
linearModelC2.fit(df_sorted_15['Ta (°C)'].to_numpy().reshape(-1, 1), df_sorted_15['Gew Abflg (mg)'])
regressionC2 = linearModelC2.predict(np.array([df_sorted_15['Ta (°C)'].max(), df_sorted_15['TM (mg)'].min()]).reshape(-1, 1))

sns.set()
custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
sns.set_theme(style='ticks', rc=custom_params)

fig0, ax0 = plt.subplots(figsize=(30, 15))
ax0.set_xlim(15, 32)
ax0.set_ylim(0, 200)

plotA1 = ax0.scatter(df_sorted_05['Ta (°C)'], df_sorted_05['Gew Ldg (mg)'], label='GewLdg 0.5 Molar', color='red', marker="o", alpha=0.5)
plotA2 = ax0.scatter(df_sorted_15['Ta (°C)'], df_sorted_15['Gew Ldg (mg)'], label='GewLdg 1.5 Molar', color='blue', marker="o", alpha=0.5)
plotA3 = ax0.plot([df_sorted_05['Ta (°C)'].max(), df_sorted_05['Ta (°C)'].min()], regressionA1, color='k', linestyle='dashed', label='LinRegress Gew Ldg 0.5 Mol')
plotA4 = ax0.plot([df_sorted_15['Ta (°C)'].max(), df_sorted_15['Ta (°C)'].min()], regressionA2, color='k', label='LinRegress Gew Ldg 1.5 Mol')


plotB1 = ax0.scatter(df_sorted_05['Ta (°C)'], df_sorted_05['TM (mg)'], label='TM 0.5 Molar', color='red', marker="^", alpha=0.5)
plotB2 = ax0.scatter(df_sorted_15['Ta (°C)'], df_sorted_15['TM (mg)'], label='TM 1.5 Molar', color='blue', marker="^", alpha=0.5)
plotB3 = ax0.plot([df_sorted_05['Ta (°C)'].max(), df_sorted_05['Ta (°C)'].min()], regressionB1, color='k', linestyle='dashed', label='LinRegress TM Ldg 0.5 Mol')
plotB4 = ax0.plot([df_sorted_15['Ta (°C)'].max(), df_sorted_15['Ta (°C)'].min()], regressionB2, color='k', label='LinRegress TM Ldg 1.5 Mol')


plotC1 = ax0.scatter(df_sorted_05['Ta (°C)'], df_sorted_05['Gew Abflg (mg)'], label='Gew Abflg 0.5 Molar', color='red', marker="s", alpha=0.5)
plotC2 = ax0.scatter(df_sorted_15['Ta (°C)'], df_sorted_15['Gew Abflg (mg)'], label='Gew Abflg 1.5 Molar', color='blue', marker="s", alpha=0.5)
plotC3 = ax0.plot([df_sorted_05['Ta (°C)'].max(), df_sorted_05['Ta (°C)'].min()], regressionC1, color='k', linestyle='dashed', label='LinRegress Gew Abflg 0.5 Mol')
plotC4 = ax0.plot([df_sorted_15['Ta (°C)'].max(), df_sorted_15['Ta (°C)'].min()], regressionC2, color='k', label='LinRegress Gew Ldg 1.5 Mol')


StringA1 = 'Linear Regression Gew Ldg' + f' 0.5 Mol:\n$R^2$ = {round(resultA1.rvalue,2)}, p-value = {calc_pval(resultA1.pvalue)}'
StringA2 = 'Linear Regression Gew Ldg' + f' 1.5 Mol:\n$R^2$ = {round(resultA2.rvalue,2)}, p-value = {calc_pval(resultA2.pvalue)}'
StringB1 = 'Linear Regression TM Ldg$' + f' 0.5 Mol:\n$R^2$ = {round(resultB1.rvalue,2)}, p-value = {calc_pval(resultB1.pvalue)}'
StringB2 = 'Linear Regression TM Ldg' + f' 1.5 Mol:\n$R^2$ = {round(resultB2.rvalue,2)}, p-value = {calc_pval(resultB2.pvalue)}'
StringC1 = 'Linear Regression Gew Abflg' + f' 0.5 Mol:\n$R^2$ = {round(resultC1.rvalue,2)}, p-value = {calc_pval(resultC1.pvalue)}'
StringC2 = 'Linear Regression Gew Abflg' + f' 1.5 Mol:\n$R^2$ = {round(resultC2.rvalue,2)}, p-value = {calc_pval(resultC2.pvalue)}'

ax0.annotate(StringA1, xy=(31, 110), fontsize=20)
ax0.annotate(StringA2, xy=(31, 98), fontsize=20)
ax0.annotate(StringB1, xy=(31, 86), fontsize=20)
ax0.annotate(StringB2, xy=(31, 74), fontsize=20)
ax0.annotate(StringC1, xy=(31, 62), fontsize=20)
ax0.annotate(StringC2, xy=(31, 50), fontsize=20)

ax0.tick_params(axis='both', which='major', labelsize=20)
ax0.legend(bbox_to_anchor=(.92, 1), loc='upper left', fontsize=20)
ax0.set_ylabel('Weight [mg]', labelpad=10, fontsize=25)
ax0.set_xlabel('T$_a$ [°C]', labelpad=10, fontsize=25)
fig0.tight_layout()

plt.savefig('GewLdg.png')