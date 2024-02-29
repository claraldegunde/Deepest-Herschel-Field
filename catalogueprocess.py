#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:06:52 2024

@author: claraaldegundemanteca
"""

import numpy as np

import pandas as pd

import sourceprocess as sp
from sourceprocess import from_m_to_F
from sourceprocess import from_hmsdms_to_deg

import astropy 
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
import math

'''
This is a module for accessing the data from the catalogues
and filtering it according to our requitements (e. g.  only with visible detections, 
below a given chi2...).
'''
#%% Import catalogues 
cat_IRAC = sp.Catalogue('IRAC')
cat_IRAC_data = cat_IRAC.get_data('/Users/claraaldegundemanteca/Desktop/Herschel Field /Chris_SPIREdarkfield/catalogues/IRACdark-matched_no_Nans.csv') # noNans.csv = nans are turned into 0


cat_XID = sp.Catalogue('XID')
cat_XID_data = cat_XID.get_data('/Users/claraaldegundemanteca/Desktop/Herschel Field /Chris_SPIREdarkfield/catalogues/XIDmultiband.csv')


#%%IRAC catalogue with optical data, only consider reliable SEDs (more points)

cat_IRAC_data_with_optical = cat_IRAC_data.query (' rmag != 99 and rmag != -1')

#Create list with indices for IRAC sources with optical data
np.save('/Users/claraaldegundemanteca/Desktop/Herschel Field /Code/list_index_optical_values.npy', np.array(cat_IRAC_data_with_optical['num'], dtype=np.int32), allow_pickle = False)


#%%IRAC catalogue with reduced chi2 less than 1

indices_red_chi2_1 = np.load('/Users/claraaldegundemanteca/Desktop/Herschel Field /Code/list_index_red_chi2<1.npy')
cat_IRAC_data_red_chi2_1 = cat_IRAC_data.iloc[indices_red_chi2_1]

indices_red_chi2_2 = np.load('/Users/claraaldegundemanteca/Desktop/Herschel Field /Code/list_index_red_chi2<2.npy')
cat_IRAC_data_red_chi2_2 = cat_IRAC_data.iloc[indices_red_chi2_2]


#%% Select data

# XID catalogue
XID_id_list = cat_XID_data['ID']
XID_redshift_list = [cat_IRAC_data['zphot'][i] for i in XID_id_list] 
XID_ra_list = cat_XID_data['RA']
XID_dec_list = cat_XID_data['Dec']

# #Full IRAC catalogue
# IRAC_redshift_list = cat_IRAC_data['zphot']
# IRAC_id_list = cat_IRAC_data['num']
# IRAC_ra_list = cat_IRAC_data['ra']
# IRAC_dec_list = cat_IRAC_data['dec']

# #Catalogue with valid optical points
# IRAC_redshift_list = cat_IRAC_data_with_optical['zphot']
# IRAC_id_list = cat_IRAC_data_with_optical['num']
# IRAC_ra_list = cat_IRAC_data_with_optical['ra']
# IRAC_dec_list = cat_IRAC_data_with_optical['dec']

#Catalogue with reduced chi2 < 1
IRAC_redshift_list = cat_IRAC_data_red_chi2_1['zphot']
IRAC_id_list = cat_IRAC_data_red_chi2_1['num']
IRAC_ra_list = cat_IRAC_data_red_chi2_1['ra']
IRAC_dec_list = cat_IRAC_data_red_chi2_1['dec']

## Catalogue with reduced chi2 < 2
# IRAC_redshift_list = cat_IRAC_data_red_chi2_2['zphot']
# IRAC_id_list = cat_IRAC_data_red_chi2_2['num']
# IRAC_ra_list = cat_IRAC_data_red_chi2_2['ra']
# IRAC_dec_list = cat_IRAC_data_red_chi2_2['dec'] 



