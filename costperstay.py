import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy


df = pd.read_excel('2022 O2_gesamt_Stud.xlsx')

df_sorted = df[['Aufenthaltsdauer', 'Ta (°C)', 'Molarität', 'Kosten/Aufenthalt (J)']]

df_sorted_05 = df_sorted[df_sorted['Molarität'] == 0.5]
df_sorted_15 = df_sorted[df_sorted['Molarität'] == 1.5]

def func(x, a, b):

    return a * np.exp(b * x)

popt_05, pcov_05 = scipy.optimize.curve_fit(
        lambda t,a,b: a*np.exp(b*t), 
        df_sorted_05['Ta (°C)'].to_numpy(), 
        df_sorted_05['Aufenthaltsdauer'].to_numpy())

popt_15, pcov_15 = scipy.optimize.curve_fit(
        lambda t,a,b: a*np.exp(b*t), 
        df_sorted_15['Ta (°C)'].to_numpy(), 
        df_sorted_15['Aufenthaltsdauer'].to_numpy())

sns.set()
custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
sns.set_theme(style='ticks', rc=custom_params)

fig0, ax0 = plt.subplots(figsize=(30, 15))

ax0.set_xlim(16, 32)
ax0.set_ylim(0, 180)

plotA1 = ax0.scatter(df_sorted_05['Ta (°C)'], df_sorted_05['Aufenthaltsdauer'], label='Duration of stay [s] 0.5 Molar', color='red', marker="o", alpha=0.5)
plotA2 = ax0.scatter(df_sorted_15['Ta (°C)'], df_sorted_15['Aufenthaltsdauer'], label='Duration of stay [s] 1.5 Molar', color='blue', marker="o", alpha=0.5)
plotA3 = plt.plot(np.arange(16, 32, 0.1), func(np.arange(16, 32, 0.1), 256.92, -0.05847), color='k', linestyle='dashed', label='Exponential Regression 0.5 Molar')
plotA4 = plt.plot(np.arange(16, 32, 0.1), func(np.arange(16, 32, 0.1), 274.95, -0.05857), color='k', label='Exponential Regression 1.5 Molar')

StringA1 = 'Exponential regression 0.5 Molar:\n$256.92 \cdot e^{-0.05847*T_a}$'
StringA2 = 'Exponential regression 1.5 Molar:\n$274.95 \cdot e^{-0.05857*T_a}$'

ax0.annotate(StringA2, xy=(28, 115), fontsize=25)
ax0.annotate(StringA1, xy=(28, 130), fontsize=25)

ax0.tick_params(axis='both', which='major', labelsize=20)
ax0.legend(loc='best', fontsize=25)
ax0.set_ylabel('Duration of stay [s]', labelpad=10, fontsize=25)
ax0.set_xlabel('T$_{a}$ [°C]', labelpad=10, fontsize=25)
fig0.tight_layout()
plt.savefig('costperstay.png')

########################################################################################################################

popt_05_2, pcov_05_2 = scipy.optimize.curve_fit(
        lambda t,a,b: a*np.exp(b*t), 
        df_sorted_05['Ta (°C)'].to_numpy(), 
        df_sorted_05['Kosten/Aufenthalt (J)'].to_numpy())

popt_15_2, pcov_15_2 = scipy.optimize.curve_fit(
        lambda t,a,b: a*np.exp(b*t), 
        df_sorted_15['Ta (°C)'].to_numpy(), 
        df_sorted_15['Kosten/Aufenthalt (J)'].to_numpy())

print(f'0.5 Mol: {popt_05_2}')
print(f'1.5 Mol: {popt_15_2}')

sns.set()
custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
sns.set_theme(style='ticks', rc=custom_params)

fig0, ax0 = plt.subplots(figsize=(30, 15))

ax0.set_xlim(16, 32)
ax0.set_ylim(0, 10)

plotA1 = ax0.scatter(df_sorted_05['Ta (°C)'], df_sorted_05['Kosten/Aufenthalt (J)'], label='Cost per stay [J] 0.5 Molar', color='red', marker="o", alpha=0.5)
plotA2 = ax0.scatter(df_sorted_15['Ta (°C)'], df_sorted_15['Kosten/Aufenthalt (J)'], label='Cost per stay [J] 1.5 Molar', color='blue', marker="o", alpha=0.5)
plotA3 = plt.plot(np.arange(16, 32, 0.1), func(np.arange(16, 32, 0.1), 23.68, -0.0843), color='k', linestyle='dashed',  label='Exponential Regression 0.5 Molar')
plotA4 = plt.plot(np.arange(16, 32, 0.1), func(np.arange(16, 32, 0.1), 16.64, -0.06355), color='k', label='Exponential Regression 1.5 Molar')

StringB1 = 'Exponential regression 0.5 Molar:\n$23.68 \cdot e^{-0.0843*T_a}$'
StringB2 = 'Exponential regression 1.5 Molar:\n$16.64 \cdot e^{-0.06355*T_a}$'

ax0.annotate(StringB1, xy=(28, 7), fontsize=25)
ax0.annotate(StringB2, xy=(28, 6), fontsize=25)

ax0.tick_params(axis='both', which='major', labelsize=20)
ax0.legend(loc='best', fontsize=25)
ax0.set_ylabel('Cost per stay [J]', labelpad=10, fontsize=25)
ax0.set_xlabel('T$_{a}$ [°C]', labelpad=10, fontsize=25)
fig0.tight_layout()
plt.savefig('costperstay_Joul.png')