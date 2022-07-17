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

df_sorted = df[['Dauer', 'Th MW', 'Molarität', 'Ca MW']]

df_sorted.dropna(inplace=True)

df_sorted_05 = df_sorted[df_sorted['Molarität'] == 0.5]
df_sorted_15 = df_sorted[df_sorted['Molarität'] == 1.5]

resultA1 = stats.linregress(df_sorted_05['Th MW'], df_sorted_05['Dauer'])
linearModelA1 = LinearRegression()
linearModelA1.fit(df_sorted_05['Th MW'].to_numpy().reshape(-1, 1), df_sorted_05['Dauer'])
regressionA1 = linearModelA1.predict(np.array([df_sorted_05['Th MW'].min(), df_sorted_05['Th MW'].max()]).reshape(-1, 1))

resultA2 = stats.linregress(df_sorted_15['Th MW'], df_sorted_15['Dauer'])
linearModelA2 = LinearRegression()
linearModelA2.fit(df_sorted_15['Th MW'].to_numpy().reshape(-1, 1), df_sorted_15['Dauer'])
regressionA2 = linearModelA2.predict(np.array([df_sorted_15['Th MW'].min(), df_sorted_15['Th MW'].max()]).reshape(-1, 1))

sns.set()
custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
sns.set_theme(style='ticks', rc=custom_params)

fig0, ax0 = plt.subplots(figsize=(30, 15))
ax0.set_xlim(34, 46)
ax0.set_ylim(0, 180)

plotA1 = ax0.scatter(df_sorted_05['Th MW'], df_sorted_05['Dauer'], label='Duration 0.5 Molar', color='red', marker="o", alpha=0.5)
plotA2 = ax0.scatter(df_sorted_15['Th MW'], df_sorted_15['Dauer'], label='Duration 1.5 Molar', color='blue', marker="o", alpha=0.5)
plotA3 = ax0.plot([df_sorted_05['Th MW'].min(), df_sorted_05['Th MW'].max()], regressionA1, color='k', linestyle='dashed', label='Linear Regression duration 0.5 Molar')
plotA4 = ax0.plot([df_sorted_15['Th MW'].min(), df_sorted_15['Th MW'].max()], regressionA2, color='k', label='Linear Regression duration 1.5 Molar')

StringA1 = 'Linear Regression' + f' 0.5 Mol:\n$R^2$ = {round(resultA1.rvalue,2)}, p-value = {calc_pval(resultA1.pvalue)}'
StringA2 = 'Linear Regression' + f' 1.5 Mol:\n$R^2$ = {round(resultA2.rvalue,2)}, p-value = {calc_pval(resultA2.pvalue)}'

ax0.annotate(StringA1, xy=(43.5, 130), fontsize=25)
ax0.annotate(StringA2, xy=(43.5, 115), fontsize=25)

ax0.tick_params(axis='both', which='major', labelsize=20)
ax0.legend(loc='best', fontsize=20)
ax0.set_ylabel('Duration of stay [s]', labelpad=10, fontsize=25)
ax0.set_xlabel('T$_{th}$ [°C]', labelpad=10, fontsize=25)
fig0.tight_layout()
plt.savefig('tmpvstemper.png')


########################################################################################################################
resultB1 = stats.linregress(df_sorted_05['Ca MW'], df_sorted_05['Dauer'])
linearModelB1 = LinearRegression()
linearModelB1.fit(df_sorted_05['Ca MW'].to_numpy().reshape(-1, 1), df_sorted_05['Dauer'])
regressionB1 = linearModelB1.predict(np.array([df_sorted_05['Ca MW'].min(), df_sorted_05['Ca MW'].max()]).reshape(-1, 1))

resultB2 = stats.linregress(df_sorted_15['Ca MW'], df_sorted_15['Dauer'])
linearModelB2 = LinearRegression()
linearModelB2.fit(df_sorted_15['Ca MW'].to_numpy().reshape(-1, 1), df_sorted_15['Dauer'])
regressionB2 = linearModelB2.predict(np.array([df_sorted_15['Ca MW'].min(), df_sorted_15['Ca MW'].max()]).reshape(-1, 1))

sns.set()
custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
sns.set_theme(style='ticks', rc=custom_params)

fig0, ax0 = plt.subplots(figsize=(30, 15))
ax0.set_xlim(24, 38)
ax0.set_ylim(0, 180)

plotA1 = ax0.scatter(df_sorted_05['Ca MW'], df_sorted_05['Dauer'], label='Duration 0.5 Molar', color='red', marker="o", alpha=0.5)
plotA2 = ax0.scatter(df_sorted_15['Ca MW'], df_sorted_15['Dauer'], label='Duration 1.5 Molar', color='blue', marker="o", alpha=0.5)
plotA3 = ax0.plot([df_sorted_05['Ca MW'].min(), df_sorted_05['Ca MW'].max()], regressionA1, color='k', linestyle='dashed', label='Linear Regression duration 0.5 Molar')
plotA4 = ax0.plot([df_sorted_15['Ca MW'].min(), df_sorted_15['Ca MW'].max()], regressionA2, color='k', label='Linear Regression duration 1.5 Molar')

StringB1 = 'Linear Regression' + f' 0.5 Mol:\n$R^2$ = {round(resultB1.rvalue,2)}, p-value = {calc_pval(resultB1.pvalue)}'
StringB2 = 'Linear Regression' + f' 1.5 Mol:\n$R^2$ = {round(resultB2.rvalue,2)}, p-value = {calc_pval(resultB2.pvalue)}'

ax0.annotate(StringB1, xy=(35, 120), fontsize=25)
ax0.annotate(StringB2, xy=(35, 105), fontsize=25)

ax0.tick_params(axis='both', which='major', labelsize=25)
ax0.legend(loc='best', fontsize=25)
ax0.set_ylabel('Duration of stay [s]', labelpad=10, fontsize=25)
ax0.set_xlabel('T$_{ca}$ [°C]', labelpad=10, fontsize=25)
fig0.tight_layout()
plt.savefig('tmpvstemper_ca.png')