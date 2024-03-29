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

#XID
cat_XID = sp.Catalogue('XID')
cat_XID.get_data('/Users/claraaldegundemanteca/Desktop/HerschelField/Chris_SPIREdarkfield/catalogues/XIDmultiband.csv')


'''
Choose which subset of the IRAC catalogue we want to work with (whole, only optical or 
reduced chi2 below some value)
'''
#Whole IRAC catalogue (original)
cat_IRAC = sp.Catalogue('IRAC whole')
cat_IRAC.get_data('/Users/claraaldegundemanteca/Desktop/HerschelField/Chris_SPIREdarkfield/catalogues/IRACdark-matched_no_nans.csv') # noNans.csv = nans are turned into 0


#Import SCUBA datafile
# cat_SCUBA = pd.read_csv('/Users/claraaldegundemanteca/Downloads/HerschelField /input_SCUBA.csv', low_memory=True)
# cat_SCUBA = pd.DataFrame(cat_SCUBA)    

# %% Create CIGALE input txt file


def CIGALE_input(ID_list, outputfile, SPIRE = True, SCUBA = False, PSW_greater_than = 0):
    '''
    Produces a dataframe containing the necessary info to perform a fit in CIGALE (id, redshift and fluxes for each filter) and converts is to a .txt file.
    Uses IRAC catalogue 
    
    Only use SCUBA  = True for the list of sources with unambiguous detections 
    
    Parameters
    ----------
    first_ID: ID to start from
    last_ID: final ID
    SPIRE = True if we want to include SPIRE points
    SCUBA = True if we want to include SCUBA points
    PSW_greater_than: If SPIRE= True, doesn't include sources whose PSW flux is less than a value
    
    Returns
    -------
    d: pandas DataFrame object
        data table containing the info to feed into CIGALE (id, redshift and fluxes for each filter)
    
    '''
    if SPIRE == False:
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
                u_err = s.fluxes['umagerr'].iloc[0]
                g = s.fluxes['gmag'].iloc[0]
                g_err = s.fluxes['gmagerr'].iloc[0]
                r = s.fluxes['rmag'].iloc[0]
                r_err = s.fluxes['rmag'].iloc[0]
                z = s.fluxes['zmagbest'].iloc[0]
                z_err = s.fluxes['zmagerrbest'].iloc[0]

        
                if type(u) == str:
                    if len(u) > 8:
                        u = -1
                if type(u_err) == str:
                    if len(u_err) > 8:
                        u_err = -1
         
                if type(g) == str:
                    if len(g) > 8:
                        g = -1                
                if type(g_err) == str:
                    if len(g_err) > 8:
                        g_err = -1
                 
                if type(r) == str:
                    if len(r) > 8:
                        r = -1
                if type(r_err) == str:
                    if len(r_err) > 8:
                        r_err = -1 
                        
                if type(z) == str:
                    if len(z) > 8:
                        z = -1
                if type(z_err) == str:
                    if len(z_err) > 8:
                        z_err = -1
                    
                u_flux = (from_m_to_F(F0, (float(u))))*1e3
                u_flux_err = u_flux *np.log(10)*float(u_err)/2.5
                g_flux = (from_m_to_F(F0, (float(g))))*1e3
                g_flux_err = g_flux *np.log(10)*float((g_err))/2.5
                r_flux = (from_m_to_F(F0, (float(r))))*1e3
                r_flux_err = r_flux *np.log(10)*float(r_err)/2.5
                z_flux = (from_m_to_F(F0, (float(z)))*1e3)
                z_flux_err = z_flux *np.log(10)*float(z_err)/2.5
                
                
                # filter out non-detections
                if float(u_err) <= 0. or float(u_err) == 99:
                    u_flux_err = 0.1*u_flux
                if float(u) <= 0. or float(u) == 99:
                    u_flux = 0.
                    u_flux_err = 0.
                if float(g_err) <= 0. or float(s.fluxes['gmagerr'].iloc[0]) == 99:
                    g_flux_err = 0.1*g_flux
                if float(g) <= 0. or float(g) == 99:
                    g_flux = 0.
                    g_flux_err = 0.
                if float(z_err) <= 0. or float(z_err) == 99:
                    z_flux_err = 0.1*z_flux
                if float(z) <= 0. or float(z) == 99:
                    z_flux = 0.
                    z_flux_err = 0.
                if float(r) <= 0. or float(r) == 99:
                    r_flux = 0.
                    r_flux_err = 0.
                if float(r_err) <= 0. or float(r_err) == 99:
                    r_flux_err = 0.1*r_flux
                
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
        sources = []
        for i in ID_list: 
            # source_i = sp.Source(id=i)
            # source_i.get_position(cat_IRAC)
            # source_i.get_fluxes(cat_IRAC)
            # sources.append(source_i)
    
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
                u_err = s.fluxes['umagerr'].iloc[0]
                g = s.fluxes['gmag'].iloc[0]
                g_err = s.fluxes['gmagerr'].iloc[0]
                r = s.fluxes['rmag'].iloc[0]
                r_err = s.fluxes['rmag'].iloc[0]
                z = s.fluxes['zmagbest'].iloc[0]
                z_err = s.fluxes['zmagerrbest'].iloc[0]

        
                if type(u) == str:
                    if len(u) > 8:
                        u = -1
                if type(u_err) == str:
                    if len(u_err) > 8:
                        u_err = -1
         
                if type(g) == str:
                    if len(g) > 8:
                        g = -1                
                if type(g_err) == str:
                    if len(g_err) > 8:
                        g_err = -1
                 
                if type(r) == str:
                    if len(r) > 8:
                        r = -1
                if type(r_err) == str:
                    if len(r_err) > 8:
                        r_err = -1 
                        
                if type(z) == str:
                    if len(z) > 8:
                        z = -1
                if type(z_err) == str:
                    if len(z_err) > 8:
                        z_err = -1
                    
                u_flux = (from_m_to_F(F0, (float(u))))*1e3
                u_flux_err = u_flux *np.log(10)*float(u_err)/2.5
                g_flux = (from_m_to_F(F0, (float(g))))*1e3
                g_flux_err = g_flux *np.log(10)*float((g_err))/2.5
                r_flux = (from_m_to_F(F0, (float(r))))*1e3
                r_flux_err = r_flux *np.log(10)*float(r_err)/2.5
                z_flux = (from_m_to_F(F0, (float(z)))*1e3)
                z_flux_err = z_flux *np.log(10)*float(z_err)/2.5
                
                
                # filter out non-detections
                if float(u_err) <= 0. or float(u_err) == 99:
                    u_flux_err = 0.1*u_flux
                if float(u) <= 0. or float(u) == 99:
                    u_flux = 0.
                    u_flux_err = 0.
                if float(g_err) <= 0. or float(s.fluxes['gmagerr'].iloc[0]) == 99:
                    g_flux_err = 0.1*g_flux
                if float(g) <= 0. or float(g) == 99:
                    g_flux = 0.
                    g_flux_err = 0.
                if float(z_err) <= 0. or float(z_err) == 99:
                    z_flux_err = 0.1*z_flux
                if float(z) <= 0. or float(z) == 99:
                    z_flux = 0.
                    z_flux_err = 0.
                if float(r) <= 0. or float(r) == 99:
                    r_flux = 0.
                    r_flux_err = 0.
                if float(r_err) <= 0. or float(r_err) == 99:
                    r_flux_err = 0.1*r_flux
                
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
                    psw_flux_err = np.sqrt((float(s.fluxes['PSW Flux Err (mJy)'].iloc[0]))**2 + 5.8**2) #Add confusion noise
                    pmw_flux = float(s.fluxes['PMW Flux (mJy)'].iloc[0])
                    pmw_flux_err =  np.sqrt((float(s.fluxes['PMW Flux Err (mJy)'].iloc[0]))**2 + 6.3**2)
                    plw_flux = float(s.fluxes['PLW Flux (mJy)'].iloc[0])
                    plw_flux_err = np.sqrt((float(s.fluxes['PLW Flux Err (mJy)'].iloc[0]))**2 + 6.8**2)
                elif len(s.fluxes.columns) == 38:
                    psw_flux = 0.0
                    psw_flux_err = 0.0
                    pmw_flux = 0.0
                    pmw_flux_err = 0.0
                    plw_flux = 0.0
                    plw_flux_err = 0.0
                else:
                    raise Exception('cannot recognise catalogues')
                
                if psw_flux > PSW_greater_than:
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
        
        if SCUBA == True: 
            df ['SCUBA850'] = cat_SCUBA['SCUBA850']
            df ['SCUBA850_err'] = cat_SCUBA['SCUBA850_err']
            df ['SCUBA450'] = cat_SCUBA['SCUBA450']
            df ['SCUBA450_err'] = cat_SCUBA['SCUBA450_err']


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
# SCUBA + SPIRE detections
# df = CIGALE_input((2705,2193,308,547,66243,742,785,851,2128,73723,1001,1065,79529,2247,2169), outputfile='/Volumes/LaCie/Clara/cigale-v2022.1/input_withSPIRE.txt', SPIRE = True, SCUBA=True)

# Whole IRAC catalogue, no SPIRE
# df = CIGALE_input(np.arange(0,86814), outputfile='/Users/claraaldegundemanteca/Downloads/cigale-v2022.1/input_noSPIRE.txt', SPIRE = False, SCUBA=False, only_valid_visible = False)


# IRAC visible sources, no SPIRE
# indices_visible = np.load('/Users/claraaldegundemanteca/Desktop/Herschel Field /Code/list_index_optical_values.npy')
# df = CIGALE_input(indices_visible, outputfile='/Users/claraaldegundemanteca/Downloads/cigale-v2022.1/input_noSPIRE_only_valid_visible.txt', SPIRE = False, SCUBA=False)


# Only visible sources IRAC catalogue, with SPIRE
# df = CIGALE_input(indices_visible, outputfile='/Users/claraaldegundemanteca/Downloads/cigale-v2022.1/input_withSPIRE_only_valid_visible.txt', SPIRE = True, SCUBA=False)

chi2_less_than = 10 
# Only consider reduced chi2 < 10, no SPIRE
# indices_noSPIRE_red_chi2_10 = np.load('/Users/claraaldegundemanteca/Desktop/Herschel Field /Code/list_index_noSPIRE_red_chi2<10.npy')
# df = CIGALE_input(indices_noSPIRE_red_chi2_10, outputfile='/Users/claraaldegundemanteca/Downloads/cigale-v2022.1/input_noSPIRE_red_chi2<10.txt', SPIRE = False, SCUBA=False)

# Only consider reduced chi2 < 10, with SPIRE
indices_withSPIRE_red_chi2_10 = np.load('/Users/claraaldegundemanteca/Desktop/HerschelField/Code/list_index_withSPIRE_red_chi2<10.npy')
# df = CIGALE_input(indices_withSPIRE_red_chi2_10, outputfile='/Users/claraaldegundemanteca/Downloads/cigale-v2022.1/input_withSPIRE_red_chi2<10.txt', SPIRE = True, SCUBA=False)


# Only consider reduced chi2 < 10, with SPIRE only PSW > 10mJy
df_PSW_greater_10mJy = CIGALE_input(indices_withSPIRE_red_chi2_10, outputfile='/Users/claraaldegundemanteca/Downloads/cigale-v2022.1/input_withSPIRE>10mJy_red_chi2<10.txt', SPIRE = True, SCUBA=False, PSW_greater_than=10)
df_PSW_greater_10mJy['redshift'] = -1
# Only consider reduced chi2 < 10, no SPIRE (only indices with PSW > 30microJy in the SPIRE=True case)
df = CIGALE_input(list(df_PSW_greater_10mJy['id']), outputfile='/Users/claraaldegundemanteca/Downloads/cigale-v2022.1/input_noSPIRE>10mJy_red_chi2<10.txt', SPIRE = False, SCUBA=False)
# Comment if we want to use the z in the catalogue
df['redshift'] = -1


# Only consider reduced chi2 < 10, with SPIRE only PSW > 30mJy
df_PSW_greater_30mJy = CIGALE_input(indices_withSPIRE_red_chi2_10, outputfile='/Users/claraaldegundemanteca/Downloads/cigale-v2022.1/input_withSPIRE>30mJy_red_chi2<10.txt', SPIRE = True, SCUBA=False, PSW_greater_than=30)
df_PSW_greater_30mJy['redshift'] = -1
# Only consider reduced chi2 < 10, no SPIRE (only indices with PSW > 30microJy in the SPIRE=True case)
df = CIGALE_input(list(df_PSW_greater_30mJy['id']), outputfile='/Users/claraaldegundemanteca/Downloads/cigale-v2022.1/input_noSPIRE>30mJy_red_chi2<10.txt', SPIRE = False, SCUBA=False)
# Comment if we want to use the z in the catalogue
df['redshift'] = -1
#%% list of interesting JWST sources

spiral = [650,596, 552,543,2302,645,1629,396 ,360]
elliptical = [17721 ,576,9614,498,8702,1422 ,1421,494]
interacting = [66815, 577,9584,1629,64632,477,2199 ,403, 62543, 28670 , 8892, 43374,414]
other = [66822, 554, 29359 ,605, 590,2388,2102,2198,2100,415,445,395, 444]

whole_ID_list = spiral + elliptical + interacting + other
#%%
df = CIGALE_input(whole_ID_list, outputfile='/Users/claraaldegundemanteca/Downloads/cigale-v2022.1/input_JWST_sources_noSPIRE.txt', SPIRE = False, SCUBA=False)
# Comment if we want to use the z in the catalogue
df['redshift'] = -1


#%% Fix SEDs

spiral = [650,596, 552,543,2302,645,1629,396 ,360, 494]
elliptical = [17721 ,576,9614,498,8702,1422 ,1421]
interacting = [66815, 577,9584,1629,64632,477,2199 ,403, 62543, 28670 , 8892, 43374,414]
other = [66822, 554, 29359 ,605, 590,2388,2102,2198,2100,415,445,395, 444]

whole_ID_list = spiral + elliptical + interacting + other

df = CIGALE_input([2303], outputfile='/Users/claraaldegundemanteca/Downloads/cigale-v2022.1/input_JWST_2303.txt', SPIRE = False, SCUBA=False)
# Comment if we want to use the z in the catalogue
df['redshift'] = -1