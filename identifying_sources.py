#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:28:01 2023

@author: claraaldegundemanteca
"""
import matplotlib.pyplot as plt
import astropy 
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
import pandas as pd 
import numpy as np

'''
This code identifies sources in the catalogue given its
coordinates in DS9

Will have to fiddle around with diff_ra and diff_dec 
for the different catalogues (different requirements)

Different functions for different catalogues because they have different ways 
of naming the columns

'''

# Useful functions

def angular_separation (ra1, ra2, dec1, dec2): #from https://www.skythisweek.info/angsep.pdf
    '''
    Returns angular separation for two sources (1 and 2) given their ra and dec
    '''
    prefactor = 180/np.pi
    numerator = np.sqrt(np.cos(dec2)*(np.sin(ra2-ra1))**2+(np.cos(dec1)*np.sin(dec2)-np.sin(dec1)*np.cos(dec2)*np.cos(ra2-ra1))**2)
    denominator = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra2-ra1)
    return prefactor * np.arctan(numerator/denominator)


def from_deg_to_hmsdms (ra, dec): #in degrees, as in the catalogue
    coordinates = SkyCoord(ra, dec, frame=FK5, unit = 'deg') # in degrees
    return coordinates.to_string('hmsdms')

#%% SELECT SOURCE

#Input coordinates of source 
ra = '17h39m45.2568s'
dec = '+68d50m15.1044s'
print('Initial coordinates:', ra, dec)

# Convert to degrees
coordinates = SkyCoord(ra, dec, frame=FK5)
ra = coordinates.ra.degree
dec = coordinates.dec.degree


#%% XID

# Import catalogue and ra and dec columns 
catalogue = pd.read_csv('/Users/claraaldegundemanteca/Desktop/Herschel Field/Chris_SPIREdarkfield/catalogues/XID_multiband.csv')
ra_catalogue = catalogue ['RA']
dec_catalogue = catalogue ['Dec']

# Find ID: need separate functions foreach catalogue because they're labelled differently
def find_source (ra_catalogue, dec_catalogue):
    '''
    Parameters
    ----------
    ra_catalogue : list, in degrees
    dec_catalogue : list, in degrees
    
    Returns
    -------
    index : index of found source (not ID)
    '''
    separation_list = []
    for i in range (0, len(ra_catalogue)):
        #scan through catalogue and record the angular separation between each entry and the coordinate we're looking for
        separation_list.append(angular_separation (ra, ra_catalogue[i], dec, dec_catalogue[i]))
    separation_array = np.array(separation_list)
    index = np .where (np.min(separation_array) == separation_array)[0]
    coordinates_in_catalogue = SkyCoord(ra_catalogue[index], dec_catalogue[index], frame=FK5, unit = 'deg') # in degrees
    # print(coordinates)
    print(ra_catalogue[index], dec_catalogue[index])
    coordinates_in_catalogue = coordinates_in_catalogue.to_string('hmsdms') # convert to hmsdms for comparison
    print('Coordinates, to compare with DS9',coordinates_in_catalogue)
    return index, catalogue['PSW Flux (mJy)'][i], catalogue['PSW Flux Err (mJy)'][i], catalogue['PMW Flux (mJy)'][i], catalogue['PMW Flux Err (mJy)'][i], catalogue['PLW Flux (mJy)'][i], catalogue['PLW Flux Err (mJy)'][i], catalogue['MIPS24 Flux (mJy)'][i], catalogue['MIPS24 Flux Err (mJy)'][i]

print(find_source (ra_catalogue, dec_catalogue))

find_source_XID = find_source(ra_catalogue, dec_catalogue)

PSW_flux_XID = find_source_XID[1]
PSW_flux_err_XID   = find_source_XID[2]
PMW_flux_XID  = find_source_XID [3]
PMW_flux_err_XID   = find_source_XID[4]
PLW_flux_XID = find_source_XID[5]
PLW_flux_err_XID = find_source_XID[6]
MIPS24_flux_XID = find_source_XID[7]
MIPS24_flux_err_XID = find_source_XID[8]

#%%  IRAC

# Import catalogue 
catalogue = pd.read_csv('/Users/claraaldegundemanteca/Desktop/Herschel Field/Chris_SPIREdarkfield/catalogues/IRACdark-matched.csv', low_memory=False)

# Select the ra and dec columns from the catalogue
ra_catalogue = catalogue ['ra']
dec_catalogue = catalogue ['dec']

# Find ID
def find_source (ra_catalogue, dec_catalogue):
    '''
    Parameters
    ----------
    ra_catalogue : list, in degrees
    dec_catalogue : list, in degrees
    
    Returns
    -------
    index : index of found source (not ID)
    '''
    separation_list = []
    for i in range (0, len(ra_catalogue)):
        #scan through catalogue and record the angular separation between each entry and the coordinate we're looking for
        separation_list.append(angular_separation (ra, ra_catalogue[i], dec, dec_catalogue[i]))
    separation_array = np.array(separation_list)
    index = np .where (np.min(separation_array) == separation_array)[0]
    coordinates_in_catalogue = SkyCoord(ra_catalogue[index], dec_catalogue[index], frame=FK5, unit = 'deg') # in degrees
    print(coordinates_in_catalogue)
    # print(coordinates)
    print(ra_catalogue[index], dec_catalogue[index])
    coordinates_in_catalogue = coordinates_in_catalogue.to_string('hmsdms') # convert to hmsdms for comparison
    print('Coordinates, to compare with DS9',coordinates_in_catalogue)
    return index,  catalogue['irac1flux'][i], catalogue['irac1fluxerr'][i], catalogue['irac2flux'][i], catalogue['irac2fluxerr'][i], catalogue['irac3flux'][i], catalogue['irac3fluxerr'][i], catalogue['irac4flux'][i], catalogue['irac4fluxerr'][i]

print(find_source (ra_catalogue, dec_catalogue))
find_source_IRAC = find_source(ra_catalogue, dec_catalogue)

# Record in format: Instrument_flux_catalogue
IRAC1_flux_IRAC =  find_source_IRAC [1]
IRAC1_flux_err_IRAC = find_source_IRAC [2]
IRAC2_flux_IRAC = find_source (ra_catalogue, dec_catalogue) [3]
IRAC2_flux_err_IRAC = find_source_IRAC [4]
IRAC3_flux_IRAC = find_source_IRAC [5]
IRAC3_flux_err_IRAC = find_source_IRAC [6]
IRAC4_flux_IRAC   = find_source_IRAC [7]
IRAC4_flux_err_IRAC = find_source_IRAC [8]
    

#%% SUSSEXtractor


# Import catalogue 
catalogue = pd.read_csv('/Users/claraaldegundemanteca/Desktop/Herschel Field/Chris_SPIREdarkfield/catalogues/SUSSEXtractor_multiband_full_singlepos.csv', low_memory=False)

# Select the ra and dec columns from the catalogue
ra_catalogue = catalogue ['RA']
dec_catalogue = catalogue ['Dec']

# Find ID
def find_source (ra_catalogue, dec_catalogue):
    '''
    Parameters
    ----------
    ra_catalogue : list, in degrees
    dec_catalogue : list, in degrees
    
    Returns
    -------
    index : index of found source (not ID)
    '''
    separation_list = []
    for i in range (0, len(ra_catalogue)):
        #scan through catalogue and record the angular separation between each entry and the coordinate we're looking for
        separation_list.append(angular_separation (ra, ra_catalogue[i], dec, dec_catalogue[i]))
    separation_array = np.array(separation_list)
    index = np .where (np.min(separation_array) == separation_array)[0]
    coordinates_in_catalogue = SkyCoord(ra_catalogue[index], dec_catalogue[index], frame=FK5, unit = 'deg') # in degrees
    # print(coordinates_in_catalogue)
    # print(coordinates)
    # print(ra_catalogue[index], dec_catalogue[index])
    coordinates_in_catalogue = coordinates_in_catalogue.to_string('hmsdms') # convert to hmsdms for comparison
    # print('Coordinates, to compare with DS9',coordinates_in_catalogue)
    return index, catalogue['PSW Flux (mJy)'][index], catalogue['PSW Flux Err (mJy)'][index], catalogue['PMW Flux (mJy)'][index], catalogue['PMW Flux Err (mJy)'][index], catalogue['PLW Flux (mJy)'][index], catalogue['PLW Flux Err (mJy)'][index] #returns index, fluxes and errors

# print(find_source (ra_catalogue, dec_catalogue))

find_source_SUSSEX = find_source(ra_catalogue, dec_catalogue)

PSW_flux_SUSSEX =  find_source_SUSSEX[1]
PSW_flux_err_SUSSEX   = find_source_SUSSEX[2]
PMW_flux_SUSSEX  = find_source_SUSSEX [3]
PMW_flux_err_SUSSEX   = find_source_SUSSEX[4]
PLW_flux_SUSSEX = find_source_SUSSEX[5]
PLW_flux_err_SUSSEX = find_source_SUSSEX[6]
    
#%% CREATE A TXT FILE

'''
filter1: PSW_flux_SUSSEX 
filter2: PMW_flux_SUSSEX
filter3: PLW_flux_SUSSEX
filter4: IRAC1_flux_IRAC
filter5: IRAC2_flux_IRAC
filter6: IRAC3_flux_IRAC
filter7: IRAC4_flux_IRAC
filter8: PSW_flux_XID
filter9: PMW_flux_XID
filter10: PLW_flux_XID
filter11: MIPS24_flux_XID
'''

d = {'filter1': PSW_flux_SUSSEX, 'filter1_err': PSW_flux_err_SUSSEX, 'filter2': PMW_flux_SUSSEX, 'filter2_err': PMW_flux_err_SUSSEX, \
     'filter3': PLW_flux_SUSSEX, 'filter3_err': PLW_flux_err_SUSSEX, 'filter4':IRAC1_flux_IRAC, 'filter4_err':IRAC1_flux_err_IRAC, \
    'filter5':IRAC2_flux_IRAC, 'filter5_err':IRAC2_flux_err_IRAC, 'filter6':IRAC3_flux_IRAC, 'filter6_err':IRAC3_flux_err_IRAC,\
    'filter7':IRAC4_flux_IRAC, 'filter7_err':IRAC4_flux_err_IRAC,'filter8': PSW_flux_XID, 'filter8_err': PSW_flux_err_XID, 'filter9': PMW_flux_XID,\
    'filter9_err': PMW_flux_err_XID, 'filter10': PLW_flux_XID, 'filter10_err': PLW_flux_err_XID, 'filter11': MIPS24_flux_XID, 'filter11_err': MIPS24_flux_err_XID}


# Create dataframe
df = pd.DataFrame(data=d)

# Convert to txt
df.to_csv(r'/Users/claraaldegundemanteca/Desktop/Herschel Field/Chris_SPIREdarkfield/CIGALE_input.txt')



