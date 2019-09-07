#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 11:43:43 2019

@author: ashish
"""


### Importing Libraries ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

dataset = pd.read_csv('financial_data.csv')


### EDA ###

dataset.head()
dataset.columns
dataset.describe()


### Cleaning the data

# Removing NaN
dataset.isna().any() 


## Histograms

dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed'])

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize = 20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(6, 5, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i -1])
    vals = np.size(dataset2.iloc[:, i -1].unique())
    if vals >= 100:
        vals = 100
    plt.hist(dataset2.iloc[:, i -1], bins = vals, color = '#3F5D7D')
    
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


## Correlation with Response Variable 
#Model like RF are not linear 

dataset2.corrwith(dataset.e_signed).plot.bar(figsize = (20, 10), title="Correlation with E-Signed", fontsize = 15, rot = 45, grid = True)


### Correaltion Matrix ###

sn.set(style="white", font_scale=2)

# Compute the correaltion matrix
corr = dataset2.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(18, 15))
f.suptitle('Correlation Matrix', fontsize = 10)

# Generate a custom diverging colormap

cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sn.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})




























































