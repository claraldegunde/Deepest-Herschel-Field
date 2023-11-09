#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:28:01 2023

@author: claraaldegundemanteca
"""
# %%
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

def from_m_to_F (F0, m):
    '''
    Given a magnitude m and zero magnitude flux F0, flux F is calculated (in the units of F0)
    '''
    return F0*10**(-m/2.5)

#%% SELECT SOURCE

#Input coordinates of source 
#ID 5
ra = '17h39m45.2568s'
dec = ' +68d50m15.072s'

print('Initial coordinates:', ra, dec)

# Convert to degrees
coordinates = SkyCoord(ra, dec, frame=FK5)
ra = coordinates.ra.degree
dec = coordinates.dec.degree


#%% SUSSEXtractor

# Import catalogue 
catalogue = pd.read_csv('/Users/claraaldegundemanteca/Desktop/HerschelField/Chris_SPIREdarkfield/catalogues/SUSSEXtractor_multiband_full_singlepos.csv', low_memory=False)

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

PSW_flux_SUSSEX =  find_source_SUSSEX[1] # all of these in mJy
PSW_flux_err_SUSSEX   = find_source_SUSSEX[2]
PMW_flux_SUSSEX  = find_source_SUSSEX [3]
PMW_flux_err_SUSSEX   = find_source_SUSSEX[4]
PLW_flux_SUSSEX = find_source_SUSSEX[5]
PLW_flux_err_SUSSEX = find_source_SUSSEX[6]
    

#%%  IRAC

# Import catalogue 
catalogue = pd.read_csv('/Users/claraaldegundemanteca/Desktop/HerschelField/Chris_SPIREdarkfield/catalogues/IRACdark-matched.csv', low_memory=False)

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
    # get redshift
    redshift = catalogue['zphot'].iloc[index]
    return index, redshift, catalogue['irac1flux'][index], catalogue['irac1fluxerr'][index], catalogue['irac2flux'][index], \
        catalogue['irac2fluxerr'][index], catalogue['irac3flux'][index], catalogue['irac3fluxerr'][index], catalogue['irac4flux'][index],\
        catalogue['irac4fluxerr'][index], catalogue['mips24flux'][index], catalogue['mips24fluxerr'][index], catalogue['mips70flux'][index],\
        catalogue['mips70fluxerr'][index], catalogue['Akariflux11'][index], catalogue['Akarierr11'][index], catalogue['Akariflux15'][index],\
        catalogue['Akarierr15'][index], catalogue['Akariflux18'][index], catalogue['Akarierr18'][index], catalogue['umag'][index], \
        catalogue['umagerr'][index], catalogue['gmag'][index],catalogue['gmagerr'][index], catalogue['rmag'][index], catalogue['rmagerr'][index],\
        catalogue['zmagbest'][index], catalogue['zmagerrbest'][index]






print('----', find_source (ra_catalogue, dec_catalogue))
find_source_IRAC = find_source(ra_catalogue, dec_catalogue)

# Record in format: Instrument_flux_catalogue
source_redshift = find_source_IRAC[1]

IRAC1_flux_IRAC =  find_source_IRAC[2]*1e-3 #convert from uJy to mJy
IRAC1_flux_err_IRAC = find_source_IRAC[3]*1e-3 #convert from uJy to mJy
IRAC2_flux_IRAC = find_source_IRAC[4]*1e-3 #convert from uJy to mJy
IRAC2_flux_err_IRAC = find_source_IRAC[5]*1e-3 #convert from uJy to mJy
IRAC3_flux_IRAC = find_source_IRAC[6]*1e-3 #convert from uJy to mJy
IRAC3_flux_err_IRAC = find_source_IRAC[7]*1e-3 #convert from uJy to mJy
IRAC4_flux_IRAC = find_source_IRAC[8]*1e-3 #convert from uJy to mJy
IRAC4_flux_err_IRAC = find_source_IRAC[9]*1e-3 #convert from uJy to mJy

MIPS24_flux_IRAC = find_source_IRAC[10] *1e-3 #convert from uJy to mJy
MIPS24_flux_err_IRAC = find_source_IRAC[11]*1e-3 #convert from uJy to mJy
MIPS70_flux_IRAC = find_source_IRAC[12]*1e-3 #convert from uJy to mJy
MIPS70_flux_err_IRAC = find_source_IRAC[13]*1e-3 #convert from uJy to mJy

S11_flux_AKARI = find_source_IRAC[14]*1e-3 #convert from uJy to mJy
S11_flux_err_AKARI = find_source_IRAC[15]*1e-3 #convert from uJy to mJy
L15_flux_AKARI = find_source_IRAC[16]*1e-3 #convert from uJy to mJy
L15_flux_err_AKARI = find_source_IRAC[17]*1e-3 #convert from uJy to mJy
L18W_flux_AKARI = find_source_IRAC[18]*1e-3 #convert from uJy to mJy
L18W_flux_err_AKARI = find_source_IRAC[19]*1e-3 #convert from uJy to mJy

#Change zero magnitude flux here
F0 = 3630 #in Jy
u_flux_IRAC = from_m_to_F(F0, float(find_source_IRAC[20]))*1e3 #convert to flux and then from Jy to mJy
u_flux_err_IRAC = u_flux_IRAC *(float(find_source_IRAC[21])/float(find_source_IRAC[20]))* (float(find_source_IRAC[21])/2.5) #convert to flux and then from Jy to mJy
g_flux_IRAC = from_m_to_F(F0, float(find_source_IRAC[22]))*1e3 #convert to flux and then from Jy to mJy
g_flux_err_IRAC = g_flux_IRAC *(float(find_source_IRAC[23])/float(find_source_IRAC[22])* (float(find_source_IRAC[23])/2.5))  #convert to flux and then from Jy to mJy
r_flux_IRAC = from_m_to_F(F0, float(find_source_IRAC[24]))*1e3 #convert to flux and then from Jy to mJy
r_flux_err_IRAC =  r_flux_IRAC *(float(find_source_IRAC[25])/float(find_source_IRAC[24]))* (float(find_source_IRAC[25])/2.5)  #convert to flux and then from Jy to mJy
z_flux_IRAC = from_m_to_F(F0, float(find_source_IRAC[26]))*1e3 #convert to flux and then from Jy to mJy
z_flux_err_IRAC = z_flux_IRAC*(float(find_source_IRAC[27])/float(find_source_IRAC[26]))* (float(find_source_IRAC[27])/2.5)  #convert to flux and then from Jy to mJy


#%% XID

# Import catalogue and ra and dec columns 
catalogue = pd.read_csv('/Users/claraaldegundemanteca/Desktop/HerschelField/Chris_SPIREdarkfield/catalogues/XID_multiband.csv')

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
    # get ID
    # source_id = catalogue['ID'][index]
    return index, catalogue['ID'][index], catalogue['PSW Flux (mJy)'][index], catalogue['PSW Flux Err (mJy)'][index], catalogue['PMW Flux (mJy)'][index], catalogue['PMW Flux Err (mJy)'][index], catalogue['PLW Flux (mJy)'][index], catalogue['PLW Flux Err (mJy)'][index], catalogue['MIPS24 Flux (mJy)'][index], catalogue['MIPS24 Flux Err (mJy)'][index]

print(find_source (ra_catalogue, dec_catalogue))

find_source_XID = find_source(ra_catalogue, dec_catalogue)

source_id = find_source_XID[1]

PSW_flux_XID = find_source_XID[2] #in mJy
PSW_flux_err_XID   = find_source_XID[3]
PMW_flux_XID  = find_source_XID [4]
PMW_flux_err_XID   = find_source_XID[5]
PLW_flux_XID = find_source_XID[6]
PLW_flux_err_XID = find_source_XID[7]

MIPS24_flux_XID = find_source_XID[8]*1e-3 #convert from uJy to mJy
MIPS24_flux_err_XID = find_source_XID[9]*1e-3 #convert from uJy to mJy

#%% CREATE A TXT FILE


# Create dataframe
df = source_id.to_frame(name='id')  # BEWARE! ID is from XID catalogue
df['redshift'] = float(source_redshift)


## Add SUSSEXtractor catalogue
#df['PSW'] = float(PSW_flux_SUSSEX)
#df['PSW_err'] = float(PSW_flux_err_SUSSEX)

#df['PMW'] = float(PMW_flux_SUSSEX)
#df['PMW_err'] = float(PMW_flux_err_SUSSEX)

#df['PLW'] = float(PLW_flux_SUSSEX)
#df['PLW_err'] = float(PLW_flux_err_SUSSEX)

# Add IRAC catalogue
if float(IRAC1_flux_IRAC) != -99. and float(IRAC1_flux_IRAC) != -1. and float(IRAC1_flux_err_IRAC) != -99. and float(IRAC1_flux_err_IRAC) != -1.:
    df['IRAC1'] = float(IRAC1_flux_IRAC)
    df['IRAC1_err'] = float(IRAC1_flux_err_IRAC)

if float(IRAC2_flux_IRAC) != -99. and float(IRAC2_flux_IRAC) != -1. and float(IRAC2_flux_err_IRAC) != -99. and float(IRAC2_flux_err_IRAC) != -1.:
    df['IRAC2'] = float(IRAC2_flux_IRAC)
    df['IRAC2_err'] = float(IRAC2_flux_err_IRAC)

if float(IRAC3_flux_IRAC) != -99. and float(IRAC3_flux_IRAC) != -1. and float(IRAC3_flux_err_IRAC) != -99. and float(IRAC3_flux_err_IRAC) != -1.:
    df['IRAC3'] = float(IRAC3_flux_IRAC)
    df['IRAC3_err'] = float(IRAC3_flux_err_IRAC)

if float(IRAC4_flux_IRAC) != -99. and float(IRAC4_flux_IRAC) != -1. and float(IRAC4_flux_err_IRAC) != -99. and float(IRAC4_flux_err_IRAC) != -1.:
    df['IRAC4'] = float(IRAC4_flux_IRAC)
    df['IRAC4_err'] = float(IRAC4_flux_err_IRAC)

#df['MIPS1'] = float(MIPS24_flux_IRAC)
#df['MIPS1_err'] = float(MIPS24_flux_err_IRAC)

# Add other filters from IRAC catalogue if available
if float(MIPS70_flux_IRAC) != -99. and float(MIPS70_flux_err_IRAC) != -99.:
    df['MIPS2'] = float(MIPS70_flux_IRAC)
    df['MIPS2_err'] = float(MIPS70_flux_err_IRAC)

if float(MIPS24_flux_IRAC) != -1. and float(MIPS24_flux_err_IRAC) != -1.:
    df['MIPS1'] = float(MIPS24_flux_IRAC)
    df['MIPS1_err'] = float(MIPS24_flux_err_IRAC)

if float(L18W_flux_AKARI) != 0. and float(L18W_flux_err_AKARI) != 0.:
    df['L18W'] = float(L18W_flux_AKARI)
    df['L18W_err'] = float(L18W_flux_err_AKARI)

if float(L15_flux_AKARI) != 0. and float(L15_flux_err_AKARI) != 0:
    df['L15'] = float(L15_flux_AKARI)
    df['L15_err'] = float(L15_flux_err_AKARI)

if float(S11_flux_AKARI) != 0. and float(S11_flux_err_AKARI) != 0:
    df['S11'] = float(S11_flux_AKARI)
    df['S11_err'] = float(S11_flux_err_AKARI)
    
# Visible
df['sdss.up'] =  float(u_flux_IRAC)
df['sdss.up_err'] =  float(u_flux_err_IRAC)
df['sdss.gp'] =  float(g_flux_IRAC)
df['sdss.gp_err'] =  float(g_flux_err_IRAC)
df['sdss.rp'] =  float(r_flux_IRAC)
df['sdss.rp_err'] =  float(r_flux_err_IRAC)
df['sdss.zp'] =  float(z_flux_IRAC)
df['sdss.zp_err'] =  float(z_flux_err_IRAC)

# df['MCam_u'] =  float(u_flux_IRAC)
# df['MCam_u_err'] =  float(u_flux_err_IRAC)
# df['MCam_g'] =  float(g_flux_IRAC)
# df['MCam_g_err'] =  float(g_flux_err_IRAC)
# df['MCam_r'] =  float(r_flux_IRAC)
# df['MCam_r_err'] =  float(r_flux_err_IRAC)
# df['MCam_z'] =  float(z_flux_IRAC)
# df['MCam_z_err'] =  float(z_flux_err_IRAC)


# Add XID catalogue
df['PSW'] =  float(PSW_flux_XID)
df['PSW_err'] =  float(PSW_flux_err_XID)

df['PMW'] =  float(PMW_flux_XID)
df['PMW_err'] =  float(PMW_flux_err_XID)

df['PLW'] =  float(PLW_flux_XID)
df['PLW_err'] =  float(PLW_flux_err_XID)

df['MIPS1'] = float(MIPS24_flux_XID)
df['MIPS1_err'] = float(MIPS24_flux_err_XID)


# Convert to txt
df.to_csv(r'/Users/claraaldegundemanteca/Desktop/CIGALE/cigale-v2022.1/input.txt', sep=' ', index=False)

# add hashtag
f = open('input.txt', 'r+')
lines = f.readlines()
newline1 = '#' + lines[0]
f.seek(0)
f.write(newline1)
f.writelines(lines[1:])
f.truncate()
f.close()

