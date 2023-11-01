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
    '''
    Converts from ra and dec (catalogues) to hmsdms (ds9)
    '''
    coordinates = SkyCoord(ra, dec, frame=FK5, unit = 'deg') # in degrees
    return coordinates.to_string('hmsdms')

#%% SELECT SOURCE

#Input coordinates of source 
ra = '17h40m31.99s'
dec = '+68d53m48.77s'
print('Initial coordinates:', ra, dec)

# Convert to degrees
coordinates = SkyCoord(ra, dec, frame=FK5)
ra = coordinates.ra.degree
dec = coordinates.dec.degree


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

print(find_source (ra_catalogue, dec_catalogue))

find_source_SUSSEX = find_source(ra_catalogue, dec_catalogue)

PSW_flux_SUSSEX =  find_source_SUSSEX[1]
PSW_flux_err_SUSSEX   = find_source_SUSSEX[2]
PMW_flux_SUSSEX  = find_source_SUSSEX [3]
PMW_flux_err_SUSSEX   = find_source_SUSSEX[4]
PLW_flux_SUSSEX = find_source_SUSSEX[5]
PLW_flux_err_SUSSEX = find_source_SUSSEX[6]
    

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
    # print(coordinates_in_catalogue)
    # print(coordinates)
    print(ra_catalogue[index], dec_catalogue[index])
    coordinates_in_catalogue = coordinates_in_catalogue.to_string('hmsdms') # convert to hmsdms for comparison
    # print('Coordinates, to compare with DS9',coordinates_in_catalogue)
    return index, catalogue['irac1flux'][index], catalogue['irac1fluxerr'][index], catalogue['irac2flux'][index], catalogue['irac2fluxerr'][index], catalogue['irac3flux'][index], catalogue['irac3fluxerr'][index], catalogue['irac4flux'][index], catalogue['irac4fluxerr'][index], catalogue['mips24flux'][index], catalogue['mips24fluxerr'][index]

print('----', find_source (ra_catalogue, dec_catalogue))
find_source_IRAC = find_source(ra_catalogue, dec_catalogue)

# Record in format: Instrument_flux_catalogue
IRAC1_flux_IRAC =  find_source_IRAC [1]
IRAC1_flux_err_IRAC = find_source_IRAC [2]
IRAC2_flux_IRAC = find_source_IRAC [3]
IRAC2_flux_err_IRAC = find_source_IRAC [4]
IRAC3_flux_IRAC = find_source_IRAC [5]
IRAC3_flux_err_IRAC = find_source_IRAC [6]
IRAC4_flux_IRAC = find_source_IRAC [7]
IRAC4_flux_err_IRAC = find_source_IRAC [8]
MIPS24_flux_IRAC = find_source_IRAC [9]
MIPS24_flux_err_IRAC = find_source_IRAC [9]

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
    # print(ra_catalogue[index], dec_catalogue[index])
    coordinates_in_catalogue = coordinates_in_catalogue.to_string('hmsdms') # convert to hmsdms for comparison
    # print('Coordinates, to compare with DS9',coordinates_in_catalogue)
    return index, catalogue['PSW Flux (mJy)'][index], catalogue['PSW Flux Err (mJy)'][index], catalogue['PMW Flux (mJy)'][index], catalogue['PMW Flux Err (mJy)'][index], catalogue['PLW Flux (mJy)'][index], catalogue['PLW Flux Err (mJy)'][index], catalogue['MIPS24 Flux (mJy)'][index], catalogue['MIPS24 Flux Err (mJy)'][index]

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

    

#%% CREATE A TXT FILE


# Create dataframe
df = PSW_flux_SUSSEX.to_frame(name="filter1")
df['filter1_err'] = float(PSW_flux_err_SUSSEX)

df['filter2'] = float(PMW_flux_SUSSEX)
df['filter2_err'] = float(PMW_flux_err_SUSSEX)

df['filter3'] = float(PLW_flux_SUSSEX)
df['filter3_err'] = float(PLW_flux_err_SUSSEX)

df['filter4'] = float(IRAC1_flux_IRAC)
df['filter4_err'] = float(IRAC1_flux_err_IRAC)

df['filter5'] = float(IRAC2_flux_IRAC)
df['filter5_err'] = float(IRAC2_flux_err_IRAC)

df['filter6'] = float(IRAC3_flux_IRAC)
df['filter6_err'] = float(IRAC3_flux_err_IRAC)

df['filter7'] = float(IRAC4_flux_IRAC)
df['filter7_err'] = float(IRAC4_flux_err_IRAC)

df['filter8'] = float(MIPS24_flux_IRAC)
df['filter8_err'] = float(MIPS24_flux_err_IRAC)

df['filter9'] =  float(PSW_flux_XID)
df['filter9_err'] =  float(PSW_flux_err_XID)

df['filter10'] =  float(PMW_flux_XID)
df['filter10_err'] =  float(PMW_flux_err_XID)

df['filter11'] =  float(PLW_flux_XID)
df['filter11_err'] =  float(PLW_flux_err_XID)

df['filter12'] = float(MIPS24_flux_XID)
df['filter12_err'] = float(MIPS24_flux_err_XID)


# Convert to txt
df.to_csv(r'/Users/claraaldegundemanteca/Desktop/Herschel Field/Chris_SPIREdarkfield/CIGALE_input.txt')



