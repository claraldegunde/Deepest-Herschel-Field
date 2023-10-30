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

def angular_separation (ra1, ra2, dec1, dec2): #from https://www.skythisweek.info/angsep.pdf
    prefactor = 180/np.pi
    numerator = np.sqrt(np.cos(dec2)*(np.sin(ra2-ra1))**2+(np.cos(dec1)*np.sin(dec2)-np.sin(dec1)*np.cos(dec2)*np.cos(ra2-ra1))**2)
    denominator = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra2-ra1)
    return prefactor * np.arctan(numerator/denominator)

#%% XID
#Input coordinates of source - THIS IS NOT THE SAME AS IN THE NEXT CODE CELLS BECAUSE THE SOURCE WE CHOSE IS NOT IN THE CATALOGUE
ra = '17h40m31s'
dec = '+68d53m48s'
print('Initial coordinates:', ra, dec)

# Convert to degrees
coordinates = SkyCoord(ra, dec, frame=FK5)
ra = coordinates.ra.degree
dec = coordinates.dec.degree

# Import catalogue and ra and dec columns 
catalogue = pd.read_csv('/Users/claraaldegundemanteca/Desktop/Herschel Field/Chris_SPIREdarkfield/catalogues/XID_multiband.csv')
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
    print(coordinates_in_catalogue)
    # print(coordinates)
    # print(ra_catalogue[index], dec_catalogue[index])
    coordinates_in_catalogue = coordinates_in_catalogue.to_string('hmsdms') # convert to hmsdms for comparison
    print('Coordinates, to compare with DS9',coordinates_in_catalogue)
    return index 

print(find_source (ra_catalogue, dec_catalogue))

#%%  IRAC

#Input coordinates of source
ra = '17h39m29s'
dec = '+68d47m51s'
print('Initial coordinates:', ra, dec)

# Convert to degrees
coordinates = SkyCoord(ra, dec, frame=FK5)
ra = coordinates.ra.degree
dec = coordinates.dec.degree

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
    # print(ra_catalogue[index], dec_catalogue[index])
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

#Input coordinates of source
ra = '17h39m29s'
dec = '+68d47m51s'
print('Initial coordinates:', ra, dec)

# Convert to degrees
coordinates = SkyCoord(ra, dec, frame=FK5)
ra = coordinates.ra.degree
dec = coordinates.dec.degree

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
    return index, catalogue['PSW Flux (mJy)'][i], catalogue['PSW Flux Err (mJy)'][i], catalogue['PMW Flux (mJy)'][i], catalogue['PMW Flux Err (mJy)'][i], catalogue['PLW Flux (mJy)'][i], catalogue['PLW Flux Err (mJy)'][i] #returns index, fluxes and errors

print(find_source (ra_catalogue, dec_catalogue))

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
'''

d = {'filter1': PSW_flux_SUSSEX, 'filter1_err': PSW_flux_err_SUSSEX, 'filter2': PMW_flux_SUSSEX, 'filter2_err': PMW_flux_err_SUSSEX, \
     'filter3': PLW_flux_SUSSEX, 'filter3_err': PLW_flux_err_SUSSEX, 'filter4':IRAC1_flux_IRAC, 'filter4_err':IRAC1_flux_err_IRAC, \
    'filter5':IRAC2_flux_IRAC, 'filter5_err':IRAC2_flux_err_IRAC, 'filter6':IRAC3_flux_IRAC, 'filter6_err':IRAC3_flux_err_IRAC,\
    'filter7':IRAC4_flux_IRAC, 'filter7_err':IRAC4_flux_err_IRAC}


# Create dataframe
df = pd.DataFrame(data=d, index=[0])

# Convert to txt
df.to_csv(r'/Users/claraaldegundemanteca/Desktop/Herschel Field/Chris_SPIREdarkfield/CIGALE_input.txt')



