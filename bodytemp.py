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


df = pd.read_excel('2022 IR_gesamt_Stud.xlsx')
df_sorted = df[['Molarität', 'Ca MW', 'Th MW', 'Ab MW', 'TaBiene MW']]

df_sorted.dropna(inplace=True)

df_sorted_05 = df_sorted[df_sorted['Molarität'] == 0.5]
df_sorted_15 = df_sorted[df_sorted['Molarität'] == 1.5]

resultA1 = stats.linregress(df_sorted_05['TaBiene MW'], df_sorted_05['Ca MW'])
linearModelA1 = LinearRegression()
linearModelA1.fit(df_sorted_05['TaBiene MW'].to_numpy().reshape(-1, 1), df_sorted_05['Ca MW'])
regressionA1 = linearModelA1.predict(np.array([df_sorted_05['TaBiene MW'].max(), df_sorted_05['TaBiene MW'].min()]).reshape(-1, 1))

resultA2 = stats.linregress(df_sorted_15['TaBiene MW'], df_sorted_15['Ca MW'])
linearModelB1 = LinearRegression()
linearModelB1.fit(df_sorted_15['TaBiene MW'].to_numpy().reshape(-1, 1), df_sorted_15['Ca MW'])
regressionB1 = linearModelB1.predict(np.array([df_sorted_15['TaBiene MW'].max(), df_sorted_15['TaBiene MW'].min()]).reshape(-1, 1))

resultB1 = stats.linregress(df_sorted_05['TaBiene MW'], df_sorted_05['Th MW'])
linearModelA2 = LinearRegression()
linearModelA2.fit(df_sorted_05['TaBiene MW'].to_numpy().reshape(-1, 1), df_sorted_05['Th MW'])
regressionA2 = linearModelA2.predict(np.array([df_sorted_05['TaBiene MW'].max(), df_sorted_05['TaBiene MW'].min()]).reshape(-1, 1))

resultB2 = stats.linregress(df_sorted_15['TaBiene MW'], df_sorted_15['Th MW'])
linearModelB2 = LinearRegression()
linearModelB2.fit(df_sorted_15['TaBiene MW'].to_numpy().reshape(-1, 1), df_sorted_15['Th MW'])
regressionB2 = linearModelB2.predict(np.array([df_sorted_15['TaBiene MW'].max(), df_sorted_15['TaBiene MW'].min()]).reshape(-1, 1))

resultC1 = stats.linregress(df_sorted_05['TaBiene MW'], df_sorted_05['Ab MW'])
linearModelA3 = LinearRegression()
linearModelA3.fit(df_sorted_05['TaBiene MW'].to_numpy().reshape(-1, 1), df_sorted_05['Ab MW'])
regressionA3 = linearModelA3.predict(np.array([df_sorted_05['TaBiene MW'].max(), df_sorted_05['TaBiene MW'].min()]).reshape(-1, 1))

resultC2 = stats.linregress(df_sorted_15['TaBiene MW'], df_sorted_15['Ab MW'])
linearModelB3 = LinearRegression()
linearModelB3.fit(df_sorted_15['TaBiene MW'].to_numpy().reshape(-1, 1), df_sorted_15['Ab MW'])
regressionB3 = linearModelB3.predict(np.array([df_sorted_15['TaBiene MW'].max(), df_sorted_15['TaBiene MW'].min()]).reshape(-1, 1))

sns.set()
custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
sns.set_theme(style='ticks', rc=custom_params)

fig0, ax0 = plt.subplots(figsize=(30, 15))
ax0.set_xlim(15, 35)
ax0.set_ylim(15, 50)

plotA1 = ax0.scatter(df_sorted_05['TaBiene MW'], df_sorted_05['Ca MW'], label='T$_c$ 0.5 Molar', color='red', marker="o", alpha=0.5)
plotA2 = ax0.scatter(df_sorted_15['TaBiene MW'], df_sorted_15['Ca MW'], label='T$_c$ 1.5 Molar', color='blue', marker="o", alpha=0.5)
plotA3 = ax0.plot([df_sorted_05['TaBiene MW'].max(), df_sorted_05['TaBiene MW'].min()], regressionA1, color='k', linestyle='dashed', label='LinearRegression T$_c$ 0.5 Molar')
plotA4 = ax0.plot([df_sorted_15['TaBiene MW'].max(), df_sorted_15['TaBiene MW'].min()], regressionB1, color='k', label='LinearRegression T$_c$ 1.5 Molar')

plotB1 = ax0.scatter(df_sorted_05['TaBiene MW'], df_sorted_05['Th MW'], label='T$_{th}$ 0.5 Molar', color='red', marker="^", alpha=0.5)
plotB2 = ax0.scatter(df_sorted_15['TaBiene MW'], df_sorted_15['Th MW'], label='T$_{th}$ 1.5 Molar', color='blue', marker="^", alpha=0.5)
plotB3 = ax0.plot([df_sorted_05['TaBiene MW'].max(), df_sorted_05['TaBiene MW'].min()], regressionA2, color='k', linestyle='dashed', label='LinearRegression T$_{th}$ 0.5 Molar')
plotB4 = ax0.plot([df_sorted_15['TaBiene MW'].max(), df_sorted_15['TaBiene MW'].min()], regressionB2, color='k', label='LinearRegression T$_{th}$ 1.5 Molar')

plotC1 = ax0.scatter(df_sorted_05['TaBiene MW'], df_sorted_05['Ab MW'], label='T$_{ab}$ 0.5 Molar', color='red', marker="s", alpha=0.5)
plotC2 = ax0.scatter(df_sorted_15['TaBiene MW'], df_sorted_15['Ab MW'], label='T$_{ab}$ 1.5 Molar', color='blue', marker="s", alpha=0.5)
plotC3 = ax0.plot([df_sorted_05['TaBiene MW'].max(), df_sorted_05['TaBiene MW'].min()], regressionA3, color='k', linestyle='dashed',  label='LinearRegression T$_{ab}$ 0.5 Molar')
plotC4 = ax0.plot([df_sorted_15['TaBiene MW'].max(), df_sorted_15['TaBiene MW'].min()], regressionB3, color='k', label='LinearRegression T$_{ab}$ 1.5 Molar')



StringA1 = 'Linear Regression $T_c$' + f' 0.5 Mol\n$R^2$ = {round(resultA1.rvalue,2)}, p-value = {calc_pval(resultA1.pvalue)}'
StringA2 = 'Linear Regression $T_c$' + f' 1.5 Mol\n$R^2$ = {round(resultA2.rvalue,2)}, p-value = {calc_pval(resultA2.pvalue)}'
StringB1 = 'Linear Regression $T_{th}$' + f' 0.5 Mol\n$R^2$ = {round(resultB1.rvalue,2)}, p-value = {calc_pval(resultB1.pvalue)}'
StringB2 = 'Linear Regression $T_{th}$' + f' 1.5 Mol\n$R^2$ = {round(resultB2.rvalue,2)}, p-value = {calc_pval(resultB2.pvalue)}'
StringC1 = 'Linear Regression $T_{ab}$' + f' 0.5 Mol\n$R^2$ = {round(resultC1.rvalue,2)}, p-value = {calc_pval(resultC1.pvalue)}'
StringC2 = 'Linear Regression $T_{ab}$' + f' 1.5 Mol\n$R^2$ = {round(resultC2.rvalue,2)}, p-value = {calc_pval(resultC2.pvalue)}'

ax0.annotate(StringA1, xy=(33, 30), fontsize=20)
ax0.annotate(StringA2, xy=(33, 28), fontsize=20)
ax0.annotate(StringB1, xy=(33, 26), fontsize=20)
ax0.annotate(StringB2, xy=(33, 24), fontsize=20)
ax0.annotate(StringC1, xy=(33, 22), fontsize=20)
ax0.annotate(StringC2, xy=(33, 20), fontsize=20)

ax0.tick_params(axis='both', which='major', labelsize=20)
ax0.legend(bbox_to_anchor=(.88, 1), loc='upper left', fontsize=20)
ax0.set_ylabel('Body temperature [°C]', labelpad=10, fontsize=25)
ax0.set_xlabel('T$_a$ [°C]', labelpad=10, fontsize=25)
fig0.tight_layout()
plt.savefig('bodytemp.png')