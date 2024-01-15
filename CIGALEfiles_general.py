#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 12:57:26 2023

@author: claraaldegundemanteca
"""

"""
@author: Clarisse Bonacina
"""

"""
A module to generate the CIGALE input files

Functions
---------

"""

# %%

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


#%% GET CATALOGUES

cat_IRAC = sp.Catalogue('IRAC')
cat_IRAC.get_data('/Users/claraaldegundemanteca/Desktop/HerschelField/Chris_SPIREdarkfield/catalogues/IRACdarkmatchednoNans.csv') # noNans.csv = nans are turned into 0
cat_XID = sp.Catalogue('XID')
cat_XID.get_data('/Users/claraaldegundemanteca/Desktop/HerschelField/Chris_SPIREdarkfield/catalogues/XIDmultiband.csv')



# %% Create CIGALE input txt file


def CIGALE_input(first_ID, last_ID, outputfile, SPIRE = True):
    '''
    Produces a dataframe containing the necessary info to perform a fit in CIGALE (id, redshift and fluxes for each filter) and converts is to a .txt file.
    Uses IRAC catalogue 
    
    Parameters
    ----------
    first_ID: ID to start from
    last_ID: final ID
    
    Returns
    -------
    d: pandas DataFrame object
        data table containing the info to feed into CIGALE (id, redshift and fluxes for each filter)
    
    '''
    if SPIRE == False: 
        ID_list = np.linspace(first_ID, last_ID, num = (last_ID - first_ID)+1)
        print(ID_list)
        sources = []
        for i in ID_list: 
            source_i = sp.Source(id=i)
            source_i.get_position(cat_IRAC)
            source_i.get_fluxes(cat_IRAC)
            sources.append(source_i)
    
        # extract data for all sources and store in a list
        data = [[] for i in range(28)] #34 = no. of points in sed + id + zphot
        
        for s in sources:
    
            if hasattr(s, 'fluxes'):    
    
                # conversions
                F0 = 3630
                
                u = s.fluxes['umag'].iloc[0]
                g = s.fluxes['gmag'].iloc[0]
                g_err = s.fluxes['gmagerr'].iloc[0]
                r = s.fluxes['rmag'].iloc[0]
                z = s.fluxes['zmagbest'].iloc[0]
        
        
                if type(u) == str:
                    if len(u) > 8:
                        u = -1
    
                
                if type(g) == str:
                    if len(g) > 8:
                        g = -1
                
                if type(g_err) == str:
                    if len(g_err) > 8:
                        g_err = -1
                 
                if type(r) == str:
                    if len(r) > 8:
                        r = -1
                
                if type(z) == str:
                    if len(z) > 8:
                        z = -1
                
                    
                u_flux = (from_m_to_F(F0, (float(u))))*1e3
                u_flux_err = u_flux *np.log(10)*float((s.fluxes['umagerr'].iloc[0]))/2.5
                g_flux = (from_m_to_F(F0, (float(g))))*1e3
                g_flux_err = g_flux *np.log(10)*float((g_err))/2.5
                r_flux = (from_m_to_F(F0, (float(r))))*1e3
                r_flux_err = r_flux *np.log(10)*float((s.fluxes['rmagerr'].iloc[0]))/2.5
                z_flux = (from_m_to_F(F0, (float(z)))*1e3)
                z_flux_err = z_flux *np.log(10)*float((s.fluxes['zmagerrbest'].iloc[0]))/2.5
                
                
                # filter out non-detections
                if float(s.fluxes['umagerr'].iloc[0]) <= 0. or float(s.fluxes['umagerr'].iloc[0]) == 99:
                    u_flux_err = 0.1*u_flux
                if float(u) <= 0. or float(u) == 99:
                    u_flux = 0.
                    u_flux_err = 0.
                if float(g_err) <= 0. or float(s.fluxes['gmagerr'].iloc[0]) == 99:
                    g_flux_err = 0.1*g_flux
                if float(g) <= 0. or float(g) == 99:
                    g_flux = 0.
                    g_flux_err = 0.
                if float(s.fluxes['zmagerrbest'].iloc[0]) <= 0. or float(s.fluxes['zmagerrbest'].iloc[0]) == 99:
                    z_flux_err = 0.1*u_flux
                if float(z) <= 0. or float(z) == 99:
                    z_flux = 0.
                    z_flux_err = 0.
                
                #Problems with MIPS70
                MIPS70  = s.fluxes['mips70flux'].iloc[0]
                if math.isnan(MIPS70): 
                    MIPS70 = 0
    
                MIPS70_err = s.fluxes['mips70fluxerr'].iloc[0]
                if math.isnan(MIPS70_err): 
                    MIPS70_err = 0
    
                entries = [s.id, #ID
                        float(s.fluxes['zphot'].iloc[0]), #REDSHIFT
                        float(s.fluxes['irac1flux'].iloc[0])*1e-3, float(s.fluxes['irac1fluxerr'].iloc[0])*1e-3, #IRAC
                        float(s.fluxes['irac2flux'].iloc[0])*1e-3, float(s.fluxes['irac2fluxerr'].iloc[0])*1e-3,
                        float(s.fluxes['irac3flux'].iloc[0])*1e-3, float(s.fluxes['irac3fluxerr'].iloc[0])*1e-3,
                        float(s.fluxes['irac4flux'].iloc[0])*1e-3, float(s.fluxes['irac4fluxerr'].iloc[0])*1e-3,
                        float(s.fluxes['mips24flux'].iloc[0])*1e-3, float(s.fluxes['mips24fluxerr'].iloc[0])*1e-3, #MIPS
                        float(MIPS70)*1e-3, float(MIPS70_err)*1e-3,
                        float(s.fluxes['Akariflux11'].iloc[0])*1e-3, float(s.fluxes['Akarierr11'].iloc[0])*1e-3, #AKARI
                        float(s.fluxes['Akariflux15'].iloc[0])*1e-3, float(s.fluxes['Akarierr15'].iloc[0])*1e-3,
                        float(s.fluxes['Akariflux18'].iloc[0])*1e-3, float(s.fluxes['Akarierr18'].iloc[0])*1e-3,
                        u_flux, u_flux_err, #U
                        g_flux, g_flux_err, #G
                        r_flux, r_flux_err, #R
                        z_flux, z_flux_err] #Z
            
                for i, col in enumerate(data):
                    col.append(entries[i])
            
            else:
                print('no fluxes found for ' + str(s))
                pass
    
        # create dataframe
        df = pd.DataFrame(list(zip(data[0], data[1], 
                                    data[2], data[3], data[4], data[5], data[6], data[7],
                                    data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15], 
                                    data[16], data[17], data[18], data[19], 
                                    data[20], data[21], data[22], data[23], data[24], data[25],
                                    data[26], data[27])),
                                    columns = ['id', 'redshift',                        
                                             'IRAC1', 'IRAC1_err', 'IRAC2', 'IRAC2_err', 'IRAC3', 'IRAC3_err', 'IRAC4', 'IRAC4_err',
                                            'MIPS1', 'MIPS1_err','MIPS2', 'MIPS2_err', 
                                            'S11', 'S11_err', 'L15', 'L15_err', 'L18W', 'L18W_err',
                                            'u_prime', 'u_prime_err', 'g_prime', 'g_prime_err', 'r_prime', 'r_prime_err', 'z_prime', 'z_prime_err'])
        
        # check for duplicates
        df_2 = df.drop_duplicates(subset=['id'], keep='first')
        
        
        # filter out non-detections
        d = df_2.replace(to_replace = [-99*1e-3, -1*1e-3, 99*1e-3], value=[0., 0., 0.]).abs()
        
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
    
    if SPIRE == True: 
        
        ID_list = np.linspace(first_ID, last_ID, num = (last_ID - first_ID)+1)
        print(ID_list)
        sources = []
        for i in ID_list: 
            source_i = sp.Source(id=i)
            source_i.get_position(cat_IRAC)
            source_i.get_fluxes(cat_IRAC)
            source_i.get_fluxes(cat_XID)
            sources.append(source_i)
            
        # extract data for all sources and store in a list
        data = [[] for i in range(34)] #34 = no. of points in sed + id + zphot
        
        for s in sources:
    
            if hasattr(s, 'fluxes'):    
    
                # conversions
                F0 = 3630
                
                u = s.fluxes['umag'].iloc[0]
                g = s.fluxes['gmag'].iloc[0]
                g_err = s.fluxes['gmagerr'].iloc[0]
                r = s.fluxes['rmag'].iloc[0]
                z = s.fluxes['zmagbest'].iloc[0]
        
        
                if type(u) == str:
                    if len(u) > 8:
                        u = -1
    
                
                if type(g) == str:
                    if len(g) > 8:
                        g = -1
                
                if type(g_err) == str:
                    if len(g_err) > 8:
                        g_err = -1
                 
                if type(r) == str:
                    if len(r) > 8:
                        r = -1
                
                if type(z) == str:
                    if len(z) > 8:
                        z = -1
                
                    
                u_flux = (from_m_to_F(F0, (float(u))))*1e3
                u_flux_err = u_flux *np.log(10)*float((s.fluxes['umagerr'].iloc[0]))/2.5
                g_flux = (from_m_to_F(F0, (float(g))))*1e3
                g_flux_err = g_flux *np.log(10)*float((g_err))/2.5
                r_flux = (from_m_to_F(F0, (float(r))))*1e3
                r_flux_err = r_flux *np.log(10)*float((s.fluxes['rmagerr'].iloc[0]))/2.5
                z_flux = (from_m_to_F(F0, (float(z)))*1e3)
                z_flux_err = z_flux *np.log(10)*float((s.fluxes['zmagerrbest'].iloc[0]))/2.5
                
                
                # filter out non-detections
                if float(s.fluxes['umagerr'].iloc[0]) <= 0. or float(s.fluxes['umagerr'].iloc[0]) == 99:
                    u_flux_err = 0.1*u_flux
                if float(u) <= 0. or float(u) == 99:
                    u_flux = 0.
                    u_flux_err = 0.
                if float(g_err) <= 0. or float(s.fluxes['gmagerr'].iloc[0]) == 99:
                    g_flux_err = 0.1*g_flux
                if float(g) <= 0. or float(g) == 99:
                    g_flux = 0.
                    g_flux_err = 0.
                if float(s.fluxes['zmagerrbest'].iloc[0]) <= 0. or float(s.fluxes['zmagerrbest'].iloc[0]) == 99:
                    z_flux_err = 0.1*u_flux
                if float(z) <= 0. or float(z) == 99:
                    z_flux = 0.
                    z_flux_err = 0.
                
                #Problems with MIPS70 having blank spaces
                MIPS70  = s.fluxes['mips70flux'].iloc[0]
                if math.isnan(MIPS70): 
                    MIPS70 = 0
    
                MIPS70_err = s.fluxes['mips70fluxerr'].iloc[0]
                if math.isnan(MIPS70_err): 
                    MIPS70_err = 0
                
                # Set SPIRE fluxes 
                if len(s.fluxes.columns) == 45:
                    psw_flux = float(s.fluxes['PSW Flux (mJy)'].iloc[0])
                    psw_flux_err = float(s.fluxes['PSW Flux Err (mJy)'].iloc[0])
                    pmw_flux = float(s.fluxes['PMW Flux (mJy)'].iloc[0])
                    pmw_flux_err =float(s.fluxes['PMW Flux Err (mJy)'].iloc[0])
                    plw_flux = float(s.fluxes['PLW Flux (mJy)'].iloc[0])
                    plw_flux_err = float(s.fluxes['PLW Flux Err (mJy)'].iloc[0])
                elif len(s.fluxes.columns) == 38:
                    psw_flux = 0.0
                    psw_flux_err = 0.0
                    pmw_flux = 0.0
                    pmw_flux_err = 0.0
                    plw_flux = 0.0
                    plw_flux_err = 0.0
                else:
                    raise Exception('cannot recognise catalogues')
    
                entries = [s.id, #ID
                        float(s.fluxes['zphot'].iloc[0]), #REDSHIFT
                        psw_flux, psw_flux_err, #SPIRE
                        pmw_flux, pmw_flux_err, 
                        plw_flux, plw_flux_err,
                        float(s.fluxes['irac1flux'].iloc[0])*1e-3, float(s.fluxes['irac1fluxerr'].iloc[0])*1e-3, #IRAC
                        float(s.fluxes['irac2flux'].iloc[0])*1e-3, float(s.fluxes['irac2fluxerr'].iloc[0])*1e-3,
                        float(s.fluxes['irac3flux'].iloc[0])*1e-3, float(s.fluxes['irac3fluxerr'].iloc[0])*1e-3,
                        float(s.fluxes['irac4flux'].iloc[0])*1e-3, float(s.fluxes['irac4fluxerr'].iloc[0])*1e-3,
                        float(s.fluxes['mips24flux'].iloc[0])*1e-3, float(s.fluxes['mips24fluxerr'].iloc[0])*1e-3, #MIPS
                        float(MIPS70)*1e-3, float(MIPS70_err)*1e-3,
                        float(s.fluxes['Akariflux11'].iloc[0])*1e-3, float(s.fluxes['Akarierr11'].iloc[0])*1e-3, #AKARI
                        float(s.fluxes['Akariflux15'].iloc[0])*1e-3, float(s.fluxes['Akarierr15'].iloc[0])*1e-3,
                        float(s.fluxes['Akariflux18'].iloc[0])*1e-3, float(s.fluxes['Akarierr18'].iloc[0])*1e-3,
                        u_flux, u_flux_err, #U
                        g_flux, g_flux_err, #G
                        r_flux, r_flux_err, #R
                        z_flux, z_flux_err] #Z
            
                for i, col in enumerate(data):
                    col.append(entries[i])
            
            else:
                print('no fluxes found for ' + str(s))
                pass



        # create dataframe
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
                                            'MIPS1', 'MIPS1_err','MIPS2', 'MIPS2_err', 
                                            'S11', 'S11_err', 'L15', 'L15_err', 'L18W', 'L18W_err',
                                            'u_prime', 'u_prime_err', 'g_prime', 'g_prime_err', 'r_prime', 'r_prime_err', 'z_prime', 'z_prime_err'])
       
        # check for duplicates
        df_2 = df.drop_duplicates(subset=['id'], keep='first')
        
        
        # filter out non-detections
        d = df_2.replace(to_replace = [-99*1e-3, -1*1e-3, 99*1e-3], value=[0., 0., 0.]).abs()
        
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



#%% Create CIGALE file for FULL list 

df = CIGALE_input(0,8000, outputfile='/Users/claraaldegundemanteca/Desktop/CIGALE/cigale-v2022.1/input_noSPIRE.txt', SPIRE = False)

# Comment if we want to use the z in the catalogue
df ['redshift'] = -1




