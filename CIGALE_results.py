#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 18:23:19 2024

@author: claraaldegundemanteca
"""
import pandas as pd 
import numpy as np

#Load data 
data = pd.read_csv('/Users/claraaldegundemanteca/Desktop/CIGALE/cigale-v2022.1/out/results.txt')

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