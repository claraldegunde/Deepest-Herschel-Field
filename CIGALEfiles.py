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

# %%

def from_m_to_F (F0, m):
    '''
    Converts magnitude to flux F (in the units of F0).

    Parameters
    ----------
    F0 : float
        zero magnitude flux
    m : float
        magnitude of the source
    
    Returns
    -------
    F : float
        flux in the units of F0

    '''

    F = F0*10**(-m/2.5)

    return F


def CIGALE_input(sources, outputfile):
    '''
    Produces a dataframe containing the necessary info to perform a fit in CIGALE (id, redshift and fluxes for each filter) and converts is to a .txt file.
    
    Parameters
    ----------
    sources : list of Source objects
        list of the sources to be include in the output file
    outputfile : str
        output file name

    Returns
    -------
    d: pandas DataFrame object
        data table containing the info to feed into CIGALE (id, redshift and fluxes for each filter)
    
    '''

    # extract data for all sources and store in a list
    data = [[] for i in range(34)]
    
    for s in sources:

        if hasattr(s, 'fluxes'):    

            # conversions
            F0 = 3630
            u_flux = (from_m_to_F(F0, (float((s.fluxes)['umag'].iloc[0]))))*1e3
            u_flux_err = u_flux * (float((s.fluxes)['umagerr'].iloc[0])/float((s.fluxes)['umag'].iloc[0])) * (float((s.fluxes)['umagerr'].iloc[0])/2.5)
            g_flux = (from_m_to_F(F0, (float((s.fluxes)['gmag'].iloc[0]))))*1e3
            g_flux_err = g_flux * (float((s.fluxes)['gmagerr'].iloc[0])/float((s.fluxes)['gmag'].iloc[0])) * (float((s.fluxes)['gmagerr'].iloc[0])/2.5)
            r_flux = (from_m_to_F(F0, (float((s.fluxes)['rmag'].iloc[0]))))*1e3
            r_flux_err = r_flux * (float((s.fluxes)['rmagerr'].iloc[0])/float((s.fluxes)['rmag'].iloc[0])) * (float((s.fluxes)['rmagerr'].iloc[0])/2.5)
            z_flux = (from_m_to_F(F0, (float((s.fluxes)['zmagbest'].iloc[0]))))*1e3
            z_flux_err = r_flux * (float((s.fluxes)['zmagerrbest'].iloc[0])/float((s.fluxes)['zmagbest'].iloc[0])) * (float((s.fluxes)['zmagerrbest'].iloc[0])/2.5)

            # filter out non-detections
            if float(s.fluxes['umagerr'].iloc[0]) <= 0. or float(s.fluxes['umagerr'].iloc[0]) == 99:
                u_flux_err = 0.1*u_flux
            if float(s.fluxes['umag'].iloc[0]) <= 0. or float(s.fluxes['umag'].iloc[0]) == 99:
                u_flux = 0.
                u_flux_err = 0.
            if float(s.fluxes['gmagerr'].iloc[0]) <= 0. or float(s.fluxes['gmagerr'].iloc[0]) == 99:
                g_flux_err = 0.1*g_flux
            if float(s.fluxes['gmag'].iloc[0]) <= 0. or float(s.fluxes['gmag'].iloc[0]) == 99:
                g_flux = 0.
                g_flux_err = 0.
            if float(s.fluxes['zmagerrbest'].iloc[0]) <= 0. or float(s.fluxes['zmagerrbest'].iloc[0]) == 99:
                z_flux_err = 0.1*u_flux
            if float(s.fluxes['zmagbest'].iloc[0]) <= 0. or float(s.fluxes['zmagbest'].iloc[0]) == 99:
                z_flux = 0.
                z_flux_err = 0.
            
            # set SPIRE fluxes 
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
                    float(s.fluxes['mips70flux'].iloc[0])*1e-3, float(s.fluxes['mips70fluxerr'].iloc[0])*1e-3,
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
                                data[26], data[27], data[28], data[29], data[30], data[31], data[32], data[33])),
                                columns = ['id', 'redshift',  
                                        'PSW', 'PSW_err', 'PMW', 'PMW_err', 'PLW', 'PLW_err',
                                        'IRAC1', 'IRAC1_err', 'IRAC2', 'IRAC2_err', 'IRAC3', 'IRAC3_err', 'IRAC4', 'IRAC4_err',
                                        'MIPS2', 'MIPS2_err', 'MIPS1', 'MIPS1_err',
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

# %%
# EXAMPLE CELL

# get catalogues
cat_IRAC = sp.Catalogue('IRAC')
cat_IRAC.get_data('catalogues/IRACdark-matched.csv')
cat_XID = sp.Catalogue('XID')
cat_XID.get_data('catalogues/XID_multiband.csv')

s3124 = sp.Source(id=3124)
s3124.get_position(cat_IRAC)


# %%
s3124.get_fluxes(cat_IRAC)

# %%
s3124.get_fluxes(cat_XID)
# %%
# get id for source at position (17h39m36.4416s, +69d04m06.924s)
# works the same in degrees
spos = sp.Source(ra='17h39m36.4416s', dec='+69d04m06.924s')
spos.get_id(cat_XID)
print(spos)

# get position for source 3
s3 = sp.Source(id=3)
s3.get_position(cat_IRAC)
print(s3)

# get fluxes for source 5
s5 = sp.Source(id=5)
s5.get_position(cat_IRAC)
print(s5)
s5.get_fluxes(cat_IRAC)
s5.get_fluxes(cat_XID)
print()
print(s5.fluxes)

# get fluxes for these 3 sources and put them into a CIGALE input file
s3.get_fluxes(cat_IRAC)
s3.get_fluxes(cat_XID)
spos.get_fluxes(cat_IRAC)
spos.get_fluxes(cat_XID)
df = CIGALE_input(sources=[spos, s3, s5], outputfile='example.txt')

# %%

# get catalogues
cat_IRAC = sp.Catalogue('IRAC')
cat_IRAC.get_data('catalogues/IRACdark-matched.csv')
cat_XID = sp.Catalogue('XID')
cat_XID.get_data('catalogues/XID_multiband.csv')

# get source
s31487 = sp.Source(id=31487)
s31487.get_position(cat_IRAC)
s31487.get_fluxes(cat_IRAC)
s31487.get_fluxes(cat_XID)

df = CIGALE_input(sources=[s31487], outputfile='31487.txt')


# %%
from sourceprocess import from_deg_to_hmsdms
c = from_deg_to_hmsdms(s3124.ra, s3124.dec)
print(c)

# %%

cat_IRAC = sp.Catalogue('IRAC')
cat_IRAC.get_data('catalogues/IRACdark-matched.csv')
cat_XID = sp.Catalogue('XID')
cat_XID.get_data('catalogues/XID_multiband.csv')

# %%
# check source 31487
s31487 = sp.Source(id=31487)
s31487.get_position(cat_IRAC)
s31487.get_fluxes(cat_IRAC)

# %%

snew = sp.Source(ra=s31487.ra, dec=s31487.dec)
snew.get_id(cat_IRAC)

# %%

snew = sp.Source(ra='17h39m36.4416s', dec='+69d04m06.924s')
snew.get_id(cat_IRAC)

# %%

id_ra = (cat_IRAC.data[cat_IRAC.columns[2] == s31487.ra])
id_dec = (cat_IRAC.data[cat_IRAC.columns[3] == s31487.dec])
# %%
