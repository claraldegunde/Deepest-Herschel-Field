#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 18:23:19 2024

@author: claraaldegundemanteca
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#%%

#Load data 
data = pd.read_csv('/Users/claraaldegundemanteca/Desktop/CIGALE/cigale-v2022.1/out/results_170124.txt')

#Create dataframe
headers = data.columns[0]
headers = headers.split(" ")
headers_list = []
for i in headers:
    # print(i)
    if i != '':
        headers_list.append(i)
   
values_list = np.zeros([len(data), len(headers_list)])
for i in range (0, len(data)): #scans through all sources
    values = data[data.columns[0]][i]
    values = values.split(" ")
    useful_values = []

    for j in values: # scans through values for the different variables
        # print(i)
        if j != '':
            useful_values.append(j)

    values_list [i, :] = useful_values
    
df = pd.DataFrame(data = values_list, columns = headers_list) #Dataframe created


#%% 

redshifts = np.array(df['bayes.universe.redshift'])
chi2 = np.array(df['best.chi_square'])

#%% impose condition on chi2

accept = (df['best.chi_square']<=0.1)
#%% z distribution

plt.hist(redshifts, bins= 20, color='k', alpha = 0.7)
plt.grid(alpha=0.7)
plt.ylabel('Frequency')
plt.xlabel('z')
plt.title('Photometric redshift distribution (from CIGALE)')

#%% chi2 distribution

plt.figure()
plt.hist(chi2, bins= 20, color='k', alpha = 0.7)
plt.grid(alpha=0.7)
plt.ylabel('Frequency')
plt.xlabel('$\chi^2$')
plt.title('$\chi^2$ distribution ')

#%% chi2 against redshift 

chi2 = np.array(chi2)
redshift = np.array(chi2)

plt.figure()
plt.scatter(redshifts[accept], chi2[accept],color='k', alpha = 0.7)
plt.grid(alpha=0.7)
plt.ylabel('$\chi^2$')
plt.xlabel('z')
plt.title('$\chi^2$ vs z ')
