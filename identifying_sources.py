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

#Input coordinates of source
ra = '17h39m29s'
dec = '+68d47m51s'
print('Initial coordinates:', ra, dec)

# Convert to degrees
coordinates = SkyCoord(ra, dec, frame=FK5)
ra = coordinates.ra.degree
dec = coordinates.dec.degree

# Import catalogue 
catalogue = pd.read_csv('/Users/claraaldegundemanteca/Desktop/Herschel Field/Chris_SPIREdarkfield/catalogues/XID_multiband.csv')

# Select the ra and dec columns from the catalogue
ra_catalogue = catalogue ['RA']
dec_catalogue = catalogue ['Dec']

# Find ID
for i in range (0, len(ra_catalogue)):
    #scan through catalogue and record the difference between each entry and the coordinate we're looking for
    diff_ra = abs (ra - ra_catalogue[i]) 
    # print(diff_ra)
    diff_dec = abs (dec - dec_catalogue[i])
    # print(diff_dec)

    if diff_ra < 0.001 and diff_dec < 0.15:
        print('Source found at ID:', catalogue['ID'][i])
        coordinates_in_catalogue = SkyCoord(ra_catalogue[i], dec_catalogue[i], frame=FK5, unit = 'deg')
        print(coordinates)
        print(ra_catalogue[i], dec_catalogue[i])
        coordinates_in_catalogue = coordinates_in_catalogue.to_string('hmsdms')
        print('Coordinates, to compare with DS9',coordinates_in_catalogue)

#%% 

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
for i in range (0, len(ra_catalogue)):
    #scan through catalogue and record the difference between each entry and the coordinate we're looking for
    diff_ra = abs (ra - ra_catalogue[i]) 
    # print(diff_ra)
    diff_dec = abs (dec - dec_catalogue[i])
    # print(diff_dec)

    if diff_ra < 0.001 and diff_dec < 0.001:
        print('Source found at ID:', catalogue['num'][i])
        coordinates_in_catalogue = SkyCoord(ra_catalogue[i], dec_catalogue[i], frame=FK5, unit = 'deg')
        print(coordinates)
        print(ra_catalogue[i], dec_catalogue[i])
        coordinates_in_catalogue = coordinates_in_catalogue.to_string('hmsdms')
        print('Coordinates, to compare with DS9',coordinates_in_catalogue)
