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

df_sorted = df[['Ta (°C)', 'Molarität', 'O2-Verbrauch/Aufenthalt (µl)', 'MW_O2-Umsatz (µl/min)']]

df_sorted_05 = df_sorted[df_sorted['Molarität'] == 0.5]
df_sorted_15 = df_sorted[df_sorted['Molarität'] == 1.5]

resultA1 = stats.linregress(df_sorted_05['Ta (°C)'], df_sorted_05['MW_O2-Umsatz (µl/min)'])
resultA2 = stats.linregress(df_sorted_15['Ta (°C)'], df_sorted_15['MW_O2-Umsatz (µl/min)'])

linearModel1 = LinearRegression()
linearModel1.fit(df_sorted_05['Ta (°C)'].to_numpy().reshape(-1, 1), df_sorted_05['MW_O2-Umsatz (µl/min)'])
regression1 = linearModel1.predict(np.array([df_sorted_05['Ta (°C)'].max(), df_sorted_05['Ta (°C)'].min()]).reshape(-1, 1))

linearModel2 = LinearRegression()
linearModel2.fit(df_sorted_15['Ta (°C)'].to_numpy().reshape(-1, 1), df_sorted_15['MW_O2-Umsatz (µl/min)'])
regression2 = linearModel2.predict(np.array([df_sorted_15['Ta (°C)'].max(), df_sorted_15['Ta (°C)'].min()]).reshape(-1, 1))

sns.set()
custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
sns.set_theme(style='ticks', rc=custom_params)

fig0, ax0 = plt.subplots(figsize=(20, 10))
ax0.set_xlim(14, 32)
ax0.set_ylim(50, 300)

plot1 = ax0.scatter(df_sorted_05['Ta (°C)'], df_sorted_05['MW_O2-Umsatz (µl/min)'], label='0.5 Molar', color='red', alpha=0.5)
plot2 = ax0.plot([df_sorted_05['Ta (°C)'].max(), df_sorted_05['Ta (°C)'].min()], regression1, color='k', linestyle='dashed', label='LinearRegression 0.5 Molar')
plot3 = ax0.scatter(df_sorted_15['Ta (°C)'], df_sorted_15['MW_O2-Umsatz (µl/min)'], label='1.5 Molar', color='blue', alpha=0.5)
plot4 = ax0.plot([df_sorted_15['Ta (°C)'].max(), df_sorted_15['Ta (°C)'].min()], regression2, color='k', label='LinearRegression 1.5 Molar')

StringA1 = 'Linear Regression' + f' 0.5 Mol:\n$R^2$ = {round(resultA1.rvalue,2)}, p-value = {calc_pval(resultA1.pvalue)}'
StringA2 = 'Linear Regression' + f' 1.5 Mol:\n$R^2$ = {round(resultA2.rvalue,2)}, p-value = {calc_pval(resultA2.pvalue)}'

ax0.annotate(StringA1, xy=(15, 280), fontsize=25)
ax0.annotate(StringA2, xy=(15, 250), fontsize=25)

ax0.legend(loc='best', fontsize=14)
ax0.tick_params(axis='both', which='major', labelsize=20)
ax0.set_ylabel('O$_2$-turnover [µl/min]', labelpad=10, fontsize=25)
ax0.set_xlabel('T$_a$ [°C]', labelpad=10, fontsize=25)
fig0.tight_layout()
plt.savefig('plot_O2.png')