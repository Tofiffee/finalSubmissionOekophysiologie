import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

df_gain = pd.read_excel('2022_Waage_gesamt_Stud.xlsx')
df_gain_sorted = df_gain[['Molarität', 'TM (mg)', 'Ta (°C)']]

df_gain_sorted_05 = df_gain_sorted[df_gain_sorted['Molarität'] == 0.5]
df_gain_sorted_15 = df_gain_sorted[df_gain_sorted['Molarität'] == 1.5]

df_gain_sorted_05['TM [ul]'] = df_gain_sorted_05['TM (mg)'].apply(lambda x: x/1.0638 if type(x) != str else 0)
df_gain_sorted_15['TM [ul]'] = df_gain_sorted_15['TM (mg)'].apply(lambda x: x/1.1919 if type(x) != str else 0)

df_gain_sorted_05['Zuckergewinn'] = df_gain_sorted_05['TM [ul]'].apply(lambda x: x*(342.296/2)/1000 if x != 0 else x)
df_gain_sorted_15['Zuckergewinn'] = df_gain_sorted_15['TM [ul]'].apply(lambda x: x*(342.296*1.5)/1000 if x != 0 else x)

df_gain_sorted_05['Brennwert'] = df_gain_sorted_05['Zuckergewinn'].apply(lambda x: x*(16.8/1000)*1000 if x != 0 else x)
df_gain_sorted_15['Brennwert'] = df_gain_sorted_15['Zuckergewinn'].apply(lambda x: x*(16.8/1000)*1000 if x != 0 else x)

df_cost = pd.read_excel('2022 O2_gesamt_Stud.xlsx')
df_cost_sorted = df_cost[['Molarität', 'Ta (°C)', "Kosten/Aufenthalt (J)"]]

df_cost_sorted_05 = df_cost_sorted[df_cost_sorted['Molarität'] == 0.5]
df_cost_sorted_15 = df_cost_sorted[df_cost_sorted['Molarität'] == 1.5]

########################################################################################################################

df_gain_sorted_05 = df_gain_sorted_05[df_gain_sorted_05['Brennwert'] > 0]
gain_05_21 = df_gain_sorted_05[df_gain_sorted_05['Ta (°C)'] <= 21]
gain_05_21_26 = df_gain_sorted_05[(df_gain_sorted_05['Ta (°C)'] > 21) & (df_gain_sorted_05['Ta (°C)'] < 26)]
gain_05_26 = df_gain_sorted_05[df_gain_sorted_05['Ta (°C)'] >= 26]

gain_05_21_mean = gain_05_21['Brennwert'].mean()
gain_05_21_26_mean = gain_05_21_26['Brennwert'].mean()
gain_05_26_mean = gain_05_26['Brennwert'].mean()

########################################################################################################################

df_gain_sorted_15 = df_gain_sorted_15[df_gain_sorted_15['Brennwert'] > 0]
gain_15_21 = df_gain_sorted_15[df_gain_sorted_15['Ta (°C)'] <= 21]
gain_15_21_26 = df_gain_sorted_15[(df_gain_sorted_15['Ta (°C)'] > 21) & (df_gain_sorted_15['Ta (°C)'] < 26)]
gain_15_26 = df_gain_sorted_15[df_gain_sorted_15['Ta (°C)'] >= 26]

gain_15_21_mean = gain_15_21['Brennwert'].mean()
gain_15_21_26_mean = gain_15_21_26['Brennwert'].mean()
gain_15_26_mean = gain_15_26['Brennwert'].mean()

########################################################################################################################

cost_05_21 = df_cost_sorted_05[df_cost_sorted_05['Ta (°C)'] <= 21]
cost_05_21_26 = df_cost_sorted_05[(df_cost_sorted_05['Ta (°C)'] > 21) & (df_cost_sorted_05['Ta (°C)'] < 26)]
cost_05_26 = df_cost_sorted_05[df_cost_sorted_05['Ta (°C)'] >= 26]

cost_05_21_mean = cost_05_21['Kosten/Aufenthalt (J)'].mean()
cost_05_21_26_mean = cost_05_21_26['Kosten/Aufenthalt (J)'].mean()
cost_05_26_mean = cost_05_26['Kosten/Aufenthalt (J)'].mean()

########################################################################################################################

cost_15_21 = df_cost_sorted_15[df_cost_sorted_15['Ta (°C)'] <= 21]
cost_15_21_26 = df_cost_sorted_15[(df_cost_sorted_15['Ta (°C)'] > 21) & (df_cost_sorted_15['Ta (°C)'] < 26)]
cost_15_26 = df_cost_sorted_15[df_cost_sorted_15['Ta (°C)'] >= 26]

cost_15_21_mean = cost_15_21['Kosten/Aufenthalt (J)'].mean()
cost_15_21_26_mean = cost_15_21_26['Kosten/Aufenthalt (J)'].mean()
cost_15_26_mean = cost_15_26['Kosten/Aufenthalt (J)'].mean()


bilanz_05_21 = gain_05_21_mean - cost_05_21_mean
bilanz_05_21_26 = gain_05_21_26_mean - cost_05_21_26_mean
bilanz_05_26 = gain_05_26_mean - cost_05_26_mean

bilanz_15_21 = gain_15_21_mean - cost_15_21_mean
bilanz_15_21_26 = gain_15_21_26_mean - cost_15_21_26_mean
bilanz_15_26 = gain_15_26_mean - cost_15_26_mean

effiancy_05_21 = (gain_05_21_mean - cost_05_21_mean)/cost_05_21_mean
effiancy_05_21_26 = (gain_05_21_26_mean - cost_05_21_26_mean)/cost_05_21_26_mean
effiancy_05_26 = (gain_05_26_mean - cost_05_26_mean)/cost_05_26_mean

effiancy_15_21 = (gain_15_21_mean - cost_15_21_mean)/cost_15_21_mean
effiancy_15_21_26 = (gain_15_21_26_mean - cost_15_21_26_mean)/cost_15_21_26_mean
effiancy_15_26 = (gain_15_26_mean - cost_15_26_mean)/cost_15_26_mean

plotA1 = [bilanz_05_21, bilanz_05_21_26, bilanz_05_26]
plotA2 = [bilanz_15_21, bilanz_15_21_26, bilanz_15_26]

plotB1 = [effiancy_05_21, effiancy_05_21_26, effiancy_05_26]
plotB2 = [effiancy_15_21, effiancy_15_21_26, effiancy_15_26]

labels = ['<=21 °C', '>21 °C-26 °C<=', '>26 °C']
x_axis = np.arange(len(labels))

sns.set()
custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
sns.set_theme(style='ticks', rc=custom_params)

WIDTH = 0.25

fig0, ax0 = plt.subplots(figsize=(20, 10))
ax0.set_ylim(0, 450)

plt.bar(x_axis - WIDTH/2, plotA1, width=WIDTH, label='0.5 Molar', edgecolor='k', color='red', alpha=0.7)
plt.bar(x_axis + WIDTH/2, plotA2, width=WIDTH, label='1.5 Molar', edgecolor='k', color='blue', alpha=0.7)

ax0.set_ylabel('Netto engery gain per stay [J]', fontsize=25)
ax0.set_xlabel('Grouped temperatures', fontsize=25)
ax0.set_xticks(ticks=x_axis, labels=labels, fontsize=20)
ax0.tick_params(axis='y', which='major', labelsize=20)
ax0.legend(bbox_to_anchor=(0.8, 1.2), loc='upper left', fontsize=20)
fig0.tight_layout()
plt.savefig('plot_energy.png')

fig1, ax1 = plt.subplots(figsize=(20, 10))
ax1.set_ylim(0, 200)

plt.bar(x_axis - WIDTH/2, plotB1, width=WIDTH, label='0.5 Molar', edgecolor='k', color='red', alpha=0.7)
plt.bar(x_axis + WIDTH/2, plotB2, width=WIDTH, label='1.5 Molar', edgecolor='k', color='blue', alpha=0.7)

ax1.set_ylabel('Netto energy efficiency [J/J]', fontsize=25)
ax1.set_xlabel('Grouped temperatures', fontsize=25)
ax1.set_xticks(ticks=x_axis, labels=labels, fontsize=20)
ax1.tick_params(axis='y', which='major', labelsize=20)
ax1.legend(bbox_to_anchor=(0.8, 1), loc='upper left', fontsize=20)
fig1.tight_layout()
plt.savefig('plot_energy_efficiency.png')


