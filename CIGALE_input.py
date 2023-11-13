"""
Created on Wed Nov 09 2023

@author: Clarisse Bonacina 
"""

# %%
# IMPORTS #

import matplotlib.pyplot as plt
import astropy 
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
import pandas as pd 
import numpy as np

# %%
# FUNCTIONS #


## MISCELLANEOUS ##

def angular_separation (ra1, ra2, dec1, dec2): # from https://www.skythisweek.info/angsep.pdf
    '''
    Returns angular separation for two sources (1 and 2) given their ra and dec
    '''
    prefactor = 180/np.pi
    numerator = np.sqrt(np.cos(dec2)*(np.sin(ra2-ra1))**2+(np.cos(dec1)*np.sin(dec2)-np.sin(dec1)*np.cos(dec2)*np.cos(ra2-ra1))**2)
    denominator = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra2-ra1)
    return prefactor * np.arctan(numerator/denominator)

def from_deg_to_hmsdms (ra, dec): # in degrees, as in the catalogue
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


## FINDING FUNCTIONS ##

def find_source_SUSSEX(catalogue, ra, dec):
    '''
    Extracts fluxes from SUSSEXtractor catalogue for a source with a given position.

    Parameters
    ----------
    catalogue : pandas dataframe
        SUSSEXtractor source catalogue
    ra : float
        ra position of the source in degrees
    dec : float
        ra position of the source in degrees
    
    Returns
    -------
    index : int
        index of source (not ID)
    fluxes : list
        fluxes and errors (PSW, PMW, PLW) in mJy

    '''
    
    # extract fluxes
    ra_catalogue = catalogue['RA']
    dec_catalogue = catalogue['Dec']
    separation_list = []
    for i in range (0, len(ra_catalogue)):
        #scan through catalogue and record the angular separation between each entry and the coordinate we're looking for
        separation_list.append(angular_separation (ra, ra_catalogue[i], dec, dec_catalogue[i]))
    separation_array = np.array(separation_list)
    index = np.where(np.min(separation_array) == separation_array)[0][0]
    fluxes = [catalogue['PSW Flux (mJy)'].iloc[index], catalogue['PSW Flux Err (mJy)'].iloc[index], 
              catalogue['PMW Flux (mJy)'].iloc[index], catalogue['PMW Flux Err (mJy)'].iloc[index], 
              catalogue['PLW Flux (mJy)'].iloc[index], catalogue['PLW Flux Err (mJy)'].iloc[index]]
    
    # check
    coordinates_in_catalogue = SkyCoord(ra_catalogue[index], dec_catalogue[index], frame=FK5, unit = 'deg') # in degrees
    coordinates_in_catalogue = coordinates_in_catalogue.to_string('hmsdms') # convert to hmsdms for comparison
    # print('Coordinates, to compare with DS9',coordinates_in_catalogue)

    return index, fluxes

def find_source_IRAC(catalogue, ra, dec):
    '''
    Extracts fluxes from IRAC dark matched catalogue for a source with a given position.

    Parameters
    ----------
    catalogue : pandas dataframe
        IRAC dark matched source catalogue
    ra : float
        ra position of the source in degrees
    dec : float
        ra position of the source in degrees
    
    Returns
    -------
    index : int
        index of source (not ID)
    redshift : float
        photometric redshift of the source
    fluxes : list
        fluxes and errors (IRAC1, IRAC2, IRAC3, IRAC4, MIPS24, MIPS70, AKARI11, AKARI15, AKARI18, U, G, R, Z) in mJy

    '''
    
    # extract fluxes
    ra_catalogue = catalogue['ra']
    dec_catalogue = catalogue['dec']
    separation_list = []
    for i in range (0, len(ra_catalogue)):
        #scan through catalogue and record the angular separation between each entry and the coordinate we're looking for
        separation_list.append(angular_separation (ra, ra_catalogue[i], dec, dec_catalogue[i]))
    separation_array = np.array(separation_list)
    index = np.where(np.min(separation_array) == separation_array)[0][0]
    redshift = catalogue['zphot'].iloc[index]
    F0 = 3630 # zero magnitude flux in mJy

    # conversions
    u_flux = (from_m_to_F(F0, float(catalogue['umag'].iloc[index])))*1e3
    u_flux_err = u_flux*(float(catalogue['umagerr'].iloc[index])/float(catalogue['umag'].iloc[index])) * (float(catalogue['umagerr'].iloc[index])/2.5)
    g_flux = (from_m_to_F(F0, float(catalogue['gmag'].iloc[index])))*1e3 
    g_flux_err = g_flux*(float(catalogue['gmagerr'].iloc[index])/float(catalogue['gmag'].iloc[index])) * (float(catalogue['gmagerr'].iloc[index])/2.5)
    r_flux = (from_m_to_F(F0, float(catalogue['rmag'].iloc[index])))*1e3
    r_flux_err = r_flux*(float(catalogue['rmagerr'].iloc[index])/float(catalogue['rmag'].iloc[index])) * (float(catalogue['rmagerr'].iloc[index])/2.5)
    z_flux = (from_m_to_F(F0, float(catalogue['zmagbest'].iloc[index])))*1e3
    z_flux_err = z_flux*(float(catalogue['zmagerrbest'].iloc[index])/float(catalogue['zmagbest'].iloc[index])) * (float(catalogue['zmagerrbest'].iloc[index])/2.5)

    fluxes = [(catalogue['irac1flux'].iloc[index])*1e-3, (catalogue['irac1fluxerr'].iloc[index])*1e-3, 
              (catalogue['irac2flux'].iloc[index])*1e-3, (catalogue['irac2fluxerr'][index])*1e-3, 
              (catalogue['irac3flux'].iloc[index])*1e-3, (catalogue['irac3fluxerr'].iloc[index])*1e-3, 
              (catalogue['irac4flux'].iloc[index])*1e-3, (catalogue['irac4fluxerr'].iloc[index])*1e-3, 
              (catalogue['mips24flux'].iloc[index])*1e-3, (catalogue['mips24fluxerr'].iloc[index])*1e-3, 
              (catalogue['mips70flux'].iloc[index])*1e-3, (catalogue['mips70fluxerr'].iloc[index])*1e-3, 
              (catalogue['Akariflux11'].iloc[index])*1e-3, (catalogue['Akarierr11'].iloc[index])*1e-3, 
              (catalogue['Akariflux15'].iloc[index])*1e-3, (catalogue['Akarierr15'].iloc[index])*1e-3, 
              (catalogue['Akariflux18'].iloc[index])*1e-3, (catalogue['Akarierr18'].iloc[index])*1e-3, 
              u_flux, u_flux_err,
              g_flux, g_flux_err,
              r_flux, r_flux_err,
              z_flux, z_flux_err]

    # check
    coordinates_in_catalogue = SkyCoord(ra_catalogue[index], dec_catalogue[index], frame=FK5, unit = 'deg') # in degrees
    coordinates_in_catalogue = coordinates_in_catalogue.to_string('hmsdms') # convert to hmsdms for comparison
    # print('Coordinates, to compare with DS9',coordinates_in_catalogue)
    
    return index, redshift, fluxes

def find_source_XID(catalogue, ra, dec):
    '''
    Extracts fluxes from XID catalogue for a source with a given position.

    Parameters
    ----------
    catalogue : pandas dataframe
        XID source catalogue
    ra : float
        ra position of the source in degrees
    dec : float
        ra position of the source in degrees
    
    Returns
    -------
    index : int
        index of source (not ID)
    id : int
        source ID in XID catalogue
    fluxes : list
        fluxes and errors (PSW, PMW, PLW, MIPS24) in mJy
    '''

    # extract fluxes
    ra_catalogue = catalogue ['RA']
    dec_catalogue = catalogue ['Dec']
    separation_list = []
    for i in range (0, len(ra_catalogue)):
        #scan through catalogue and record the angular separation between each entry and the coordinate we're looking for
        separation_list.append(angular_separation (ra, ra_catalogue[i], dec, dec_catalogue[i]))
    separation_array = np.array(separation_list)
    index = np .where (np.min(separation_array) == separation_array)[0][0]
    id = catalogue['ID'].iloc[index]
    fluxes = [catalogue['PSW Flux (mJy)'].iloc[index], catalogue['PSW Flux Err (mJy)'].iloc[index],
              catalogue['PMW Flux (mJy)'].iloc[index], catalogue['PMW Flux Err (mJy)'].iloc[index], 
              catalogue['PLW Flux (mJy)'].iloc[index], catalogue['PLW Flux Err (mJy)'].iloc[index], 
              catalogue['MIPS24 Flux (mJy)'].iloc[index], catalogue['MIPS24 Flux Err (mJy)'].iloc[index]]

    # check
    coordinates_in_catalogue = SkyCoord(ra_catalogue[index], dec_catalogue[index], frame=FK5, unit = 'deg') # in degrees
    coordinates_in_catalogue = coordinates_in_catalogue.to_string('hmsdms') # convert to hmsdms for comparison
    # print('Coordinates, to compare with DS9',coordinates_in_catalogue)
 
    return index, id, fluxes

## CREATE TXT FILE ##

def CIGALE_input(cat_SUSSEX_path, cat_IRAC_path, cat_XID_path, sources, outputfile):
    '''
    Produces a dataframe containing the necessary info to perform a fit in CIGALE (id, redshift and fluxes for each filter) and converts is to a .txt file.
    
    Parameters
    ----------
    cat_SUSSEX_path : string
        path to SUSSEXtrcator catalogue (.csv file)
    cat_IRAC_path : string
        path to IRAC dark-matched catalogue (.csv file)
    cat_XID_path : string
        path to XID multiband catalogue (.csv file)
    sources : 2 x N list of strings
        ra-dec positions of the sources to retrieve fluxes from in hhmmss
    outputfile : string
        name of the output file (.txt file)

    Returns
    -------
    d: pandas DataFrame object
        data table containing the info to feed into CIGALE (id, redshift and fluxes for each filter)
    '''

    # import catalogues
    SUSSEX = pd.read_csv(cat_SUSSEX_path, low_memory=False)
    IRAC = pd.read_csv(cat_IRAC_path, low_memory=False)
    XID = pd.read_csv(cat_XID_path, low_memory=False)

    # extract data and store in a list
    data = [[] for i in range(34)]
    for coords in sources:
        ra_ini = coords[0]
        dec_ini = coords[1]
        print(ra_ini)
        coordinates = SkyCoord(ra_ini, dec_ini, frame=FK5)
        ra = coordinates.ra.degree
        dec = coordinates.dec.degree
        source_SUSSEX = find_source_SUSSEX(SUSSEX, ra, dec)
        source_IRAC = find_source_IRAC(IRAC, ra, dec)
        source_XID = find_source_XID(XID, ra, dec)
        entries = [source_XID[1], #ID
                   source_IRAC[1], #REDSHIFT
                   source_XID[2][0], source_XID[2][1], #SPIRE
                   source_XID[2][2], source_XID[2][3], 
                   source_XID[2][4], source_XID[2][5],
                   source_IRAC[2][0], source_IRAC[2][1], #IRAC
                   source_IRAC[2][2], source_IRAC[2][3],
                   source_IRAC[2][4], source_IRAC[2][5],
                   source_IRAC[2][6], source_IRAC[2][7],
                   source_IRAC[2][8], source_IRAC[2][9], #MIPS
                   source_IRAC[2][10], source_IRAC[2][11],
                   source_IRAC[2][12], source_IRAC[2][13], #AKARI
                   source_IRAC[2][14], source_IRAC[2][15],
                   source_IRAC[2][16], source_IRAC[2][17],
                   source_IRAC[2][18], source_IRAC[2][19], #U
                   source_IRAC[2][20], source_IRAC[2][21], #G
                   source_IRAC[2][22], source_IRAC[2][23], #R
                   source_IRAC[2][24], source_IRAC[2][25]] #Z
        for i, col in enumerate(data):
            col.append(entries[i])

    # create dataframe
    df = pd.DataFrame(list(zip(data[0], data[1], 
                                   data[2], data[3], data[4], data[5], data[6], data[7],
                                   data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15], 
                                   data[16], data[17], data[18], data[19], 
                                   data[20], data[21], data[22], data[23], data[24], data[25],
                                   data[26], data[27], data[28], data[29], data[30], data[31], data[32], data[33])),
                             columns = ['id', 'redshift',  
                                        'PSW', 'PSW_err', 'PMW', 'PMW_err', 'PLW', 'PLW_err',
                                        'IRAC1', 'IRAC1_err', 'IRAC2', 'IRAC2_err', 'IRAC3', 'IRAC3_err', 'IRAC4', 'IRAC4_err',
                                        'MIPS2', 'MIPS2_err', 'MIPS1', 'MIPS1_err',
                                        'S11', 'S11_err', 'L15', 'L15_err', 'L18W', 'L18W_err',
                                        'MCam_u', 'MCam_u_err', 'MCam_g', 'MCam_g_err', 'MCam_r', 'MCam_r_err', 'MCam_z', 'MCam_z_err'])
    
    # check for duplicates
    df_2 = df.drop_duplicates(subset=['id'], keep='first')

    # filter out non-detections
    d = df_2.replace(to_replace = [-99*1e-3, -1*1e-3, 99*1e-3], value = [0., 0., 0.]).abs()
    
    # Convert to txt and add hashtag
    d.to_csv(outputfile, sep=' ', index=False)
    f = open(outputfile, 'r+')
    lines = f.readlines()
    newline1 = '#' + lines[0]
    f.seek(0)
    f.write(newline1)
    f.writelines(lines[1:])
    f.truncate()
    f.close()
    
    return d

# %%

# TEST MAKE FILE
positions = [['17h39m45.26s', '+68d50m15.07s'],
             ['17h40m31.99s', '+68d53m48.77s'],
             ['17h40m12.06s', '+69d00m18.18s'],
             ['17h37m42.53s', '+68d40m36.21s'],
             ['17h43m19.85s', '+68d45m42.32s'],
             ['17h41m28.00s', '+68d57m28.18s'],
             ['17h43m08.39s', '+69d00m31.13s'],
             ['17h39m21.58s', '+69d07m36.93s'],
             ['17h38m38.41s', '+69d16m32.03s'],
             ['17h43m17.33s', '+69d19m38.11s'],
             ['17h38m02.19s', '+69d02m58.27s'],
             ['17h37m29.56s', '+69d12m18.29s'],
             ['17h36m41.95s', '+68d41m04.39s'],
             ['17h41m05.24s', '+69d10m11.45s'],
             ['17h37m52.30s', '+68d54m39.51s']]

sources = CIGALE_input('data\spiredarkfield_2023-10-11\SPIREdarkfield\catalogues\SUSSEXtractor_multiband_full_singlepos.csv',
                       'data\spiredarkfield_2023-10-11\SPIREdarkfield\catalogues\IRACdark-matched.csv',
                       'data\spiredarkfield_2023-10-11\SPIREdarkfield\catalogues\XID_multiband.csv',
                        positions, 'brightsources.txt')

# %%
