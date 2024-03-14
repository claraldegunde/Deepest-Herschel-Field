#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:37:50 2024

@author: claraaldegundemanteca
"""
import numpy as np


import matplotlib.pyplot as plt
#plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams["font.size"] = 14
import pandas as pd

import aplpy

from astropy.io import fits
from astropy.modeling import models

import sourceprocess as sp
from astropy.wcs import WCS
import petrofit as pf
from astropy.nddata import CCDData
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
import time
from matplotlib.ticker import MultipleLocator

# import webbpsf

plt.rcParams['image.cmap'] = 'gist_heat'
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#E69F00",'#762900', "#56B4E9", "#009E73",  '#882255', "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"])
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#332288", "#44AA99", "#DDCC77",  '#CC6677', "#88CCEE", "#882255", "#117733"]) 
plt.rcParams['font.serif'] = "Arial"
plt.rcParams['font.family'] = "Sans serif"
plt.rcParams['font.size'] = 20
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

#%% Only shorter wavelengths + JWST

def cross_id_plot_6 (ra = False, dec = False, ID = False):
    '''
    Given ra and dec or ID, produces a plot of the cross identifications of 
    JWST with IRAC (1-4) and MIPS
    '''
    
    fig = plt.figure(figsize=(20,15))
    plt.rcParams['font.serif'] = "Arial"
    plt.rcParams['font.family'] = "Sans serif"
    plt.rcParams['font.size'] = 18
    
    rad = 0.001
        
    if ra != False and dec != False:

        fig1 = aplpy.FITSFigure('/Users/claraaldegundemanteca/Desktop/HerschelField/JWST/jw02738-o005_t002_nircam_clear-f200w_i2d.fits', figure=fig, subplot=[0.2,0.6,0.25,0.35], north = True)
        fig1.recenter(x=ra, y=dec, radius=rad)
        fig1.show_colorscale(vmin = 0.1, vmax = 20, stretch = 'log', cmap='gist_heat')
        fig1.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
        fig1.tick_labels.hide_x()
        fig1.axis_labels.hide_x()
        fig1.axis_labels.set_ytext('Dec (ICRS)')
        
        
        fig2 = aplpy.FITSFigure('IRAC/IRAC1.fits', figure=fig, subplot=[0.45,0.6,0.25, 0.35], label='s')
        fig2.recenter(x=ra, y=dec, radius=rad)
        fig2.show_colorscale(vmin = 0.01, vmax = 15, stretch = 'log', cmap='gist_heat')
        fig2.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
        fig2.tick_labels.hide_x()
        fig2.axis_labels.hide_x()
        fig2.tick_labels.hide_y()
        fig2.axis_labels.hide_y()
        fig2.axis_labels.set_ytext('Dec (ICRS)')
        
        
        fig3 = aplpy.FITSFigure('IRAC/IRAC2.fits', figure=fig, subplot=[0.7,0.6,0.25, 0.35])
        fig3.recenter(x=ra, y=dec, radius=rad)
        fig3.show_colorscale(vmin = 0.01, vmax = 15, stretch = 'log', cmap='gist_heat')
        fig3.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
        fig3.tick_labels.hide_x()
        fig3.axis_labels.hide_x()
        fig3.tick_labels.hide_y()
        fig3.axis_labels.hide_y()
        
        
        fig4 = aplpy.FITSFigure('IRAC/IRAC3.fits', figure=fig, subplot=[0.2,0.25,0.25, 0.35])
        fig4.recenter(x=ra, y=dec, radius=rad)
        fig4.show_colorscale(vmin = 0.01, vmax = 15, stretch = 'log', cmap='gist_heat')
        fig4.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
        
        fig5 = aplpy.FITSFigure('IRAC/IRAC4.fits', figure=fig, subplot=[0.45,0.25,0.25, 0.35])
        fig5.recenter(x=ra, y=dec, radius=rad)
        fig5.show_colorscale(vmin = 0.01, vmax = 15, stretch = 'log', cmap='gist_heat')
        fig5.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
        fig5.tick_labels.hide_y()
        fig5.axis_labels.hide_y()
        
        
        fig6 = aplpy.FITSFigure('MIPS/MIPS24.fits', figure=fig, subplot=[0.7,0.25,0.25, 0.35])
        fig6.recenter(x=ra, y=dec, radius=rad)
        fig6.show_colorscale(cmap='gist_heat')
        fig6.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
        fig6.tick_labels.hide_y()
        fig6.axis_labels.hide_y()
        
        
        
        fig.show()
        # fig.savefig('/Users/claraaldegundemanteca/Desktop/Herschel Field /Code/JSWT_crossid_plot_tiral.png', dpi=300)
        fig.savefig('/Users/claraaldegundemanteca/Desktop/HerschelField/Code/JSWT_crossid_plot_577.png', dpi=300)
        
    else:
        source = Source(id = ID)
        cat_IRAC = sp.Catalogue('IRAC')
        cat_IRAC.get_data('/Users/claraaldegundemanteca/Desktop/HerschelField/Chris_SPIREdarkfield/catalogues/IRACdark-matched_no_nans.csv') # noNans.csv = nans are turned into 0
        source.get_position(cat_IRAC)
        ra = source.ra
        dec = source.dec 
        
        
        fig1 = aplpy.FITSFigure('/Users/claraaldegundemanteca/Desktop/HerschelField/JWST/jw02738-o005_t002_nircam_clear-f200w_i2d.fits', figure=fig, subplot=[0.2,0.6,0.25,0.35])
        fig1.recenter(x=ra, y=dec, radius=rad)
        fig1.show_colorscale(vmin = 0.1, vmax = 20, stretch = 'log', cmap='gist_heat')
        fig1.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
        fig1.tick_labels.hide_x()
        fig1.axis_labels.hide_x()
        fig1.axis_labels.set_ytext('Dec (ICRS)')
        
        
        fig2 = aplpy.FITSFigure('IRAC/IRAC1.fits', figure=fig, subplot=[0.45,0.6,0.25, 0.35], label='s')
        fig2.recenter(x=ra, y=dec, radius=rad)
        fig2.show_colorscale(vmin = 0.01, vmax = 15, stretch = 'log', cmap='gist_heat')
        fig2.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
        fig2.tick_labels.hide_x()
        fig2.axis_labels.hide_x()
        fig2.tick_labels.hide_y()
        fig2.axis_labels.hide_y()
        fig2.axis_labels.set_ytext('Dec (ICRS)')
        
        
        fig3 = aplpy.FITSFigure('IRAC/IRAC2.fits', figure=fig, subplot=[0.7,0.6,0.25, 0.35])
        fig3.recenter(x=ra, y=dec, radius=rad)
        fig3.show_colorscale(vmin = 0.01, vmax = 15, stretch = 'log', cmap='gist_heat')
        fig3.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
        fig3.tick_labels.hide_x()
        fig3.axis_labels.hide_x()
        fig3.tick_labels.hide_y()
        fig3.axis_labels.hide_y()
        
        
        fig4 = aplpy.FITSFigure('IRAC/IRAC3.fits', figure=fig, subplot=[0.2,0.25,0.25, 0.35])
        fig4.recenter(x=ra, y=dec, radius=rad)
        fig4.show_colorscale(vmin = 0.01, vmax = 15, stretch = 'log', cmap='gist_heat')
        fig4.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
        
        fig5 = aplpy.FITSFigure('IRAC/IRAC4.fits', figure=fig, subplot=[0.45,0.25,0.25, 0.35])
        fig5.recenter(x=ra, y=dec, radius=rad)
        fig5.show_colorscale(vmin = 0.01, vmax = 15, stretch = 'log', cmap='gist_heat')
        fig5.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
        fig5.tick_labels.hide_y()
        fig5.axis_labels.hide_y()
        
        
        fig6 = aplpy.FITSFigure('MIPS/MIPS24.fits', figure=fig, subplot=[0.7,0.25,0.25, 0.35])
        fig6.recenter(x=ra, y=dec, radius=rad)
        fig6.show_colorscale(cmap='gist_heat')
        fig6.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
        fig6.tick_labels.hide_y()
        fig6.axis_labels.hide_y()
        
        
        
        fig.show()
        fig.savefig('/Users/claraaldegundemanteca/Desktop/HerschelField/Code/JSWT_crossid_plot_%s.png'% (str(ID)), dpi=300)
        
#%% Get JWST PSF

#Run this in ipythonconsole if it doesnt work
#import os
#os.environ['WEBBPSF_PATH'] = '/Users/claraaldegundemanteca/Desktop/HerschelField/Code/webbpsf-data' 
nc = webbpsf.NIRCam()
nc.filter =  'F200W'
psf = nc.calc_psf(oversample=4)     # returns an astropy.io.fits.HDUlist containing PSF and header
plt.imshow(psf[0].data)             # display it on screen yourself, or
webbpsf.display_psf(psf)            # use this convenient function to make a nice log plot with labeled axes
nc.calc_psf("JWST_PSF.fits")

#%% 
def sersic_model_JWST (ID, radius, input_fits, A0 = 0.1, r_eff0 = 20, n0 = 4, x0 = 100, y0 = 100, ellip0 = 0.8, theta0 = 0.5, plot_image = False):
    '''
    Parameters
    ----------
    ID : ID of source
    radius : Radius IN PIXELS
    input_fits : input fits file (JWST or HST)
    output_fits :output fits file
    initial guesses :NEED TO GIVE THEM AS ACCURATE AS POSSIBBLE
    SO IT DOESNT TAKE LONG TO RUN. COMPARE MODEL FOR DIFFERENT INITIAL GUESSES
    UNTIL WE'RE HAPPY :)'
        A0, r_eff0, n0, x0, y0, ellip0, theta0 
    
    Returns: Fit parameters
    --------
    Will plot a cutout of the region of the sky we're fitting to, just to check
    List of parameters and list of errors (standard deviations using covariance matrix)
    [0] Amplitude: Brightness 
    [1] r_eff: not sure about units! 
    [2] n:  Sérsic index (larger for elliptical (4-10), smaller for spiral). 
    [3] ellipticityf: 0 - 1 (how elliptical it is, 1 = ellipse)
    [4] theta: in radians, ange of rotation wrt positive x axis 
    '''
    source = Source(id = ID)
    cat_IRAC = sp.Catalogue('IRAC')
    cat_IRAC.get_data('/Users/claraaldegundemanteca/Desktop/HerschelField/Chris_SPIREdarkfield/catalogues/IRACdark-matched_no_nans.csv') # noNans.csv = nans are turned into 0
    source.get_position(cat_IRAC)
    ra = source.ra
    dec = source.dec 
    
    # Open the FITS file
    hdul = fits.open('/Users/claraaldegundemanteca/Desktop/HerschelField/JWST/jw02738-o005_t002_nircam_clear-f200w_i2d.fits')

    data = hdul[1].data
    header = hdul[1].header

    # Get the WCS information
    wcs = WCS(header)
    pixel_scale_deg = wcs.pixel_scale_matrix[0, 0]  # Assuming square pixels, use [0, 0] for X axis
    # Convert pixel scale from degrees to arcseconds
    pixel_scale_arcsec = pixel_scale_deg * 3600 #1 pisexl is this number of arcsecs
    
    # Convert RA, Dec to pixel coordinates
    x, y = wcs.all_world2pix(ra, dec, 0)
    # Determine bounding box for the crop area
    size = 2 * radius
    crop_box = ((y - size/2), (y + size/2), (x - size/2), (x + size/2))
    
    # Crop the image data
    cropped_data = data[int(crop_box[0]):int(crop_box[1]), int(crop_box[2]):int(crop_box[3])]

    # Update FITS header if necessary (for example, if you want to update WCS)
    # header.update(...)  # Update header as needed

    # Create a new FITS HDU with the cropped data and updated header
    hdu = fits.PrimaryHDU(data=cropped_data, header=header)
    # Save the cropped FITS file
    output_fits = ('/Users/claraaldegundemanteca/Desktop/HerschelField/Code/cutout_%s.fits'% str(ID))
    hdu.writeto(output_fits, overwrite=True)
    
    #Plot image to check we're seeing the right source
    if plot_image == True: 
        fig_source = aplpy.FITSFigure(output_fits)
        fig_source.show_colorscale(vmin = 0.1, vmax = 20, stretch = 'log', cmap='gist_heat')

    #Image to find Sersic index
    hdulist = fits.open(output_fits)
    image =  hdulist[0].data
    
    #Normalise image
    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val)    
    
    # Load PSF
    PSF = fits.getdata('/Users/claraaldegundemanteca/Desktop/HerschelField/Code/JWST_PSF.fits')
    PSF = PSF / PSF.sum() #normalise it

    #Initial guesses for the fit
    amplitude= A0
    r_eff= r_eff0
    n= n0
    x_0=radius
    y_0=radius
    ellip= ellip0
    theta= theta0

    center_slack = 20
    
    #Create model
    sersic_model = models.Sersic2D(
            amplitude=amplitude,
            r_eff=r_eff,
            n=n,
            x_0=x_0,
            y_0=y_0,
            ellip=ellip,
            theta=theta,
            bounds = pf.get_default_sersic_bounds({
                'x_0': (x_0 - center_slack/2, x_0 + center_slack/2),
                'y_0': (y_0 - center_slack/2, y_0 + center_slack/2),
            }))
            
            
    #Model for PSF
    psf_sersic_model = pf.PSFConvolvedModel2D(sersic_model, psf=PSF, oversample=4, psf_oversample=1)

    #Perform the fit
    fitting_weights = None
    fitted_model, fit_info = pf.fit_model(
        image, psf_sersic_model,
        weights=fitting_weights,
        calc_uncertainties=True,
        maxiter=10000,
        epsilon=1.4901161193847656e-08,
        acc=1e-09)
    
    fitted_model.cov_matrix
    param_stds = fitted_model.stds
    print(param_stds)
    #Plot fit 
    pf.plot_fit(fitted_model, image.data, figsize=[3*8, 3])
    #Produce list of params and list of errors
    plt.savefig('/Users/claraaldegundemanteca/Desktop/HerschelField/Code/Sérsic Models/Sérsic_model_%s.png'% (str(ID)), dpi = 300)
        
    list_params = [fitted_model.amplitude.value, fitted_model.r_eff.value, fitted_model.n.value, fitted_model.ellip.value, fitted_model.theta.value]
    list_std = param_stds.stds
    print(list_std)
    
    return list_params, list_std[:3] + list_std[5:7]  #without including x_0 and y_0

#%%

input_fits = '/Users/claraaldegundemanteca/Desktop/HerschelField/JWST/jw02738-o005_t002_nircam_clear-f200w_i2d.fits'
# %%

a = sersic_model(ID=17721, radius = 100, input_fits = input_fits, A0 = 0.1, r_eff0 = 20, n0 = 2, x0 = 100, y0 = 100, ellip0 = 0.1, theta0 = 0.5 )

#%%

sersic_df = pd.DataFrame(columns=['ID','Amplitude','r_eff', 'n', 'Ellipticity', 'Theta', 
                                  'Amplitude_std','r_eff_std', 'n_std', 'Ellipticity_std', 'Theta_std'])

for ID in whole_ID_list:
    print('Source ID:', ID)
    params = sersic_model(ID=ID, radius = 100, input_fits = input_fits, A0 = 0.1, r_eff0 = 20, n0 = 2, x0 = 100, y0 = 100, ellip0 = 0.1, theta0 = 0.5 )
    model_df = pd.DataFrame([[ID, params[0][0],params[0][1], params[0][2], params[0][3], params[0][4], 
                              params[1][1],params[1][1], params[1][2], params[1][3], params[1][4]]],
                        columns=['ID','Amplitude','r_eff', 'n', 'Ellipticity', 'Theta', 
                                'Amplitude_std','r_eff_std', 'n_std', 'Ellipticity_std', 'Theta_std'])
    sersic_df  = pd.concat([sersic_df , model_df])
    
#%%
sersic_df_2 = pd.DataFrame(columns=['ID','Amplitude','r_eff', 'n', 'Ellipticity', 'Theta', 
                                  'Amplitude_std','r_eff_std', 'n_std', 'Ellipticity_std', 'Theta_std'])

for ID in elliptical[1:]:
    print('Source ID:', ID)
    params = sersic_model(ID=ID, radius = 100, input_fits = input_fits, A0 = 0.1, r_eff0 = 20, n0 = 2, x0 = 100, y0 = 100, ellip0 = 0.1, theta0 = 0.5 )
    model_df = pd.DataFrame([[ID, params[0][0],params[0][1], params[0][2], params[0][3], params[0][4], 
                              params[1][1],params[1][1], params[1][2], params[1][3], params[1][4]]],
                        columns=['ID','Amplitude','r_eff', 'n', 'Ellipticity', 'Theta', 
                                'Amplitude_std','r_eff_std', 'n_std', 'Ellipticity_std', 'Theta_std'])
    sersic_df_2  = pd.concat([sersic_df_2 , model_df])
    

#%%
sersic_df_3 = pd.DataFrame(columns=['ID','Amplitude','r_eff', 'n', 'Ellipticity', 'Theta', 
                                  'Amplitude_std','r_eff_std', 'n_std', 'Ellipticity_std', 'Theta_std'])

for ID in interacting:
    print('Source ID:', ID)
    params = sersic_model(ID=ID, radius = 100, input_fits = input_fits, A0 = 0.1, r_eff0 = 20, n0 = 2, x0 = 100, y0 = 100, ellip0 = 0.1, theta0 = 0.5 )
    model_df = pd.DataFrame([[ID, params[0][0],params[0][1], params[0][2], params[0][3], params[0][4], 
                              params[1][1],params[1][1], params[1][2], params[1][3], params[1][4]]],
                        columns=['ID','Amplitude','r_eff', 'n', 'Ellipticity', 'Theta', 
                                'Amplitude_std','r_eff_std', 'n_std', 'Ellipticity_std', 'Theta_std'])
    sersic_df_3  = pd.concat([sersic_df_3 , model_df])

#%%
sersic_df_4 = pd.DataFrame(columns=['ID','Amplitude','r_eff', 'n', 'Ellipticity', 'Theta', 
                                  'Amplitude_std','r_eff_std', 'n_std', 'Ellipticity_std', 'Theta_std'])

for ID in [8892,414]:
    print('Source ID:', ID)
    params = sersic_model(ID=ID, radius = 100, input_fits = input_fits, A0 = 0.1, r_eff0 = 20, n0 = 2, x0 = 100, y0 = 100, ellip0 = 0.1, theta0 = 0.5 )
    model_df = pd.DataFrame([[ID, params[0][0],params[0][1], params[0][2], params[0][3], params[0][4], 
                              params[1][1],params[1][1], params[1][2], params[1][3], params[1][4]]],
                        columns=['ID','Amplitude','r_eff', 'n', 'Ellipticity', 'Theta', 
                                'Amplitude_std','r_eff_std', 'n_std', 'Ellipticity_std', 'Theta_std'])
    sersic_df_4  = pd.concat([sersic_df_4 , model_df])

#%%
sersic_df_5 = pd.DataFrame(columns=['ID','Amplitude','r_eff', 'n', 'Ellipticity', 'Theta', 
                                  'Amplitude_std','r_eff_std', 'n_std', 'Ellipticity_std', 'Theta_std'])

for ID in  [554 ,605, 590,2388,2102,2198,2100,415,445,395, 444]:
    print('Source ID:', ID)
    params = sersic_model(ID=ID, radius = 100, input_fits = input_fits, A0 = 0.1, r_eff0 = 20, n0 = 2, x0 = 100, y0 = 100, ellip0 = 0.1, theta0 = 0.5 )
    model_df = pd.DataFrame([[ID, params[0][0],params[0][1], params[0][2], params[0][3], params[0][4], 
                              params[1][1],params[1][1], params[1][2], params[1][3], params[1][4]]],
                        columns=['ID','Amplitude','r_eff', 'n', 'Ellipticity', 'Theta', 
                                'Amplitude_std','r_eff_std', 'n_std', 'Ellipticity_std', 'Theta_std'])
    sersic_df_5  = pd.concat([sersic_df_5 , model_df])


#%% Put together all dataframes 

total_sersic_df =   pd.concat([sersic_df , sersic_df_2, sersic_df_3, sersic_df_4, sersic_df_5])
total_sersic_df.to_csv('sersic_parameters.txt', sep='\t', index=False)

#%% n distribution 

fig, axs = plt.subplots(1,1, figsize=(12,12))

axs.hist(total_sersic_df['n'], bins = np.linspace(0, 10, 10),   edgecolor ='k', label= '', alpha = 0.7)
axs.tick_params(direction='in',top=True,right=True,which='both')
axs.xaxis.set_minor_locator(MultipleLocator(0.5))
axs.yaxis.set_minor_locator(MultipleLocator(1))
axs.xaxis.set_major_locator(MultipleLocator(1))
# axs.grid(alpha=0.3, which = 'both')
axs.legend(facecolor = 'k', framealpha = 0.1, loc='upper left')
axs.spines['bottom'] 
axs.spines['top'] 
axs.spines['right'] 
axs.spines['left'] 
axs.set_xlabel('log SFR $(M_{\odot}/ yr)$')
axs.set_ylabel('Frequency')


fig.savefig('SFR_distribution.png', dpi = 300, transparent=True)



#%% 8 wavelengths + JWST

ra = 265.03729	
dec = 68.98705

fig = plt.figure(figsize=(20,12))
plt.rcParams['font.serif'] = "Arial"
plt.rcParams['font.family'] = "Sans serif"
plt.rcParams['font.size'] = 16

rad = 0.0015

fig1 = aplpy.FITSFigure('/Users/claraaldegundemanteca/Desktop/HerschelField/JWST/jw02738-o005_t002_nircam_clear-f200w_i2d.fits', figure=fig, subplot=[0.2,0.7,0.25,0.25])
fig1.show_colorscale(vmin = 0.08, vmax = 20, stretch = 'log', cmap='gist_heat')
fig1.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
fig1.tick_labels.hide_x()
fig1.axis_labels.hide_x()
fig1.axis_labels.set_ytext('Dec (ICRS)')

fig2 = aplpy.FITSFigure('IRAC/IRAC1.fits', figure=fig, subplot=[0.45,0.7,0.25, 0.25], label='s')
fig2.recenter(x=ra, y=dec, radius=rad)
fig2.show_colorscale(vmin = 0.01, vmax = 15, stretch = 'log', cmap='gist_heat')
fig2.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
fig2.tick_labels.hide_x()
fig2.axis_labels.hide_x()
fig2.tick_labels.hide_y()
fig2.axis_labels.hide_y()
fig2.axis_labels.set_ytext('Dec (ICRS)')

fig3 = aplpy.FITSFigure('IRAC/IRAC2.fits', figure=fig, subplot=[0.7,0.7,0.25, 0.25])
fig3.recenter(x=ra, y=dec, radius=rad)
fig3.show_colorscale(vmin = 0.01, vmax = 15, stretch = 'log', cmap='gist_heat')
fig3.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
fig3.tick_labels.hide_x()
fig3.axis_labels.hide_x()
fig3.tick_labels.hide_y()
fig3.axis_labels.hide_y()

fig4 = aplpy.FITSFigure('IRAC/IRAC3.fits', figure=fig, subplot=[0.2,0.45,0.25, 0.25])
fig4.recenter(x=ra, y=dec, radius=rad)
fig4.show_colorscale(vmin = 0.01, vmax = 15, stretch = 'log', cmap='gist_heat')
fig4.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
fig4.tick_labels.hide_x()
fig4.axis_labels.hide_x()

fig5 = aplpy.FITSFigure('IRAC/IRAC4.fits', figure=fig, subplot=[0.45,0.45,0.25, 0.25])
fig5.recenter(x=ra, y=dec, radius=rad)
fig5.show_colorscale(vmin = 0.01, vmax = 15, stretch = 'log', cmap='gist_heat')
fig5.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
fig5.tick_labels.hide()
fig5.axis_labels.hide()

fig6 = aplpy.FITSFigure('MIPS/MIPS24.fits', figure=fig, subplot=[0.7,0.45,0.25, 0.25])
fig6.recenter(x=ra, y=dec, radius=rad)
fig6.show_colorscale(cmap='gist_heat')
fig6.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
fig6.tick_labels.hide()
fig6.axis_labels.hide()

fig7 = aplpy.FITSFigure('SPIRE/PSW_masked.fits', figure=fig, subplot=[0.2,0.20,0.25, 0.25])
fig7.recenter(x=ra, y=dec, radius=rad)
fig7.show_colorscale(cmap='gist_heat')
fig7.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
fig7.axis_labels.set_ytext('Dec (ICRS)')
fig7.axis_labels.set_xtext('RA (ICRS)')
plt.annotate(text='HST',xy=(0.8,0.2),xycoords='figure fraction', zorder=0)

fig8 = aplpy.FITSFigure('SPIRE/PMW_masked.fits', figure=fig, subplot=[0.45,0.20,0.25, 0.25])
fig8.recenter(x=ra, y=dec, radius=rad)
fig8.show_colorscale(cmap='gist_heat')
fig8.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
fig8.axis_labels.set_xtext('RA (ICRS)')
fig8.tick_labels.hide_y()
fig8.axis_labels.hide_y()

fig9 = aplpy.FITSFigure('SPIRE/PLW_masked.fits', figure=fig, subplot=[0.7,0.20,0.25, 0.25])
fig9.recenter(x=ra, y=dec, radius=rad)
fig9.show_colorscale(cmap='gist_heat')
fig9.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.1, 0.25, 1], overlap=True, linewidth=0.3, alpha=0.5)
fig9.tick_labels.hide_y()
fig9.axis_labels.hide_y()
fig9.tick_labels.set_xposition('bottom')
fig9.axis_labels.set_xtext('RA (ICRS)')

fig.show()
fig.savefig('/Users/claraaldegundemanteca/Desktop/Herschel Field /Code/JSWT_crossid_plot_tiral.png', dpi=300)


#%% IDs from JWST

# cross_id_plot_6 (ra = ra, dec = dec)
# cross_id_plot_6 (ID = 577)


spiral = [650,596, 552,543,2303,645,1629,396 ,360, 494]
elliptical = [17721 ,576,9614,498,8702,1422 ,1421]
interacting = [66815, 577,9584,1629,64632,477,2199 ,403, 62543, 28670 , 8892, 43374,414]
other = [66822, 554, 29359 ,605, 590,2388,2102,2198,2100,415,445,395, 444]

whole_ID_list = spiral + elliptical + interacting + other

# %% Plot all JWST sources with 6 wavelengths

# for i in whole_ID_list:
#     cross_id_plot_6 (ID = i)

for i in whole_ID_list:
    cross_id_plot_6 (ID = i)


#%% Sérsic models for HST

HST_index_list = [3124, 2007,2257,2367,2010,2252,1258,1533,1265,84804,16450,2036,2030,81201,2251,2612,2008,2006]
input_fits = '/Users/claraaldegundemanteca/Desktop/HerschelField/HST/25704051_HST.fits'

sersic_df = pd.DataFrame(columns=['ID','Amplitude','r_eff', 'n', 'Ellipticity', 'Theta', 
                                  'Amplitude_std','r_eff_std', 'n_std', 'Ellipticity_std', 'Theta_std'])

for ID in HST_index_list:
    print('Source ID:', ID)
    params = sersic_model(ID=ID, radius = 100, input_fits = input_fits, A0 = 0.1, r_eff0 = 20, n0 = 2, x0 = 100, y0 = 100, ellip0 = 0.1, theta0 = 0.5 )
    model_df = pd.DataFrame([[ID, params[0][0],params[0][1], params[0][2], params[0][3], params[0][4], 
                              params[1][1],params[1][1], params[1][2], params[1][3], params[1][4]]],
                        columns=['ID','Amplitude','r_eff', 'n', 'Ellipticity', 'Theta', 
                                'Amplitude_std','r_eff_std', 'n_std', 'Ellipticity_std', 'Theta_std'])
    sersic_df  = pd.concat([sersic_df , model_df])
#%%

def sersic_model_HST (ID, radius, input_fits, A0 = 0.1, r_eff0 = 20, n0 = 4, x0 = 100, y0 = 100, ellip0 = 0.8, theta0 = 0.5, plot_image = False):
    '''
    Parameters
    ----------
    ID : ID of source
    radius : Radius IN PIXELS
    input_fits : input fits file (JWST or HST)
    output_fits :output fits file
    initial guesses :NEED TO GIVE THEM AS ACCURATE AS POSSIBBLE
    SO IT DOESNT TAKE LONG TO RUN. COMPARE MODEL FOR DIFFERENT INITIAL GUESSES
    UNTIL WE'RE HAPPY :)'
        A0, r_eff0, n0, x0, y0, ellip0, theta0 
    
    Returns: Fit parameters
    --------
    Will plot a cutout of the region of the sky we're fitting to, just to check
    List of parameters and list of errors (standard deviations using covariance matrix)
    [0] Amplitude: Brightness 
    [1] r_eff: not sure about units! 
    [2] n:  Sérsic index (larger for elliptical (4-10), smaller for spiral). 
    [3] ellipticityf: 0 - 1 (how elliptical it is, 1 = ellipse)
    [4] theta: in radians, ange of rotation wrt positive x axis 
    '''
    source = Source(id = ID)
    cat_IRAC = sp.Catalogue('IRAC')
    cat_IRAC.get_data('/Users/claraaldegundemanteca/Desktop/HerschelField/Chris_SPIREdarkfield/catalogues/IRACdark-matched_no_nans.csv') # noNans.csv = nans are turned into 0
    source.get_position(cat_IRAC)
    ra = source.ra
    dec = source.dec 
    
    # Open the FITS file
    hdul = fits.open(input_fits)
    data = hdul[1].data
    header = hdul[1].header

    # Get the WCS information
    wcs = WCS(header)
    pixel_scale_deg = wcs.pixel_scale_matrix[0, 0]  # Assuming square pixels, use [0, 0] for X axis
    # Convert pixel scale from degrees to arcseconds
    pixel_scale_arcsec = pixel_scale_deg * 3600 #1 pisexl is this number of arcsecs
    
    # Convert RA, Dec to pixel coordinates
    x, y = wcs.all_world2pix(ra, dec, 0)
    # Determine bounding box for the crop area
    size = 2 * radius
    crop_box = ((y - size/2), (y + size/2), (x - size/2), (x + size/2))
    
    # Crop the image data
    cropped_data = data[int(crop_box[0]):int(crop_box[1]), int(crop_box[2]):int(crop_box[3])]

    # Update FITS header if necessary (for example, if you want to update WCS)
    # header.update(...)  # Update header as needed

    # Create a new FITS HDU with the cropped data and updated header
    hdu = fits.PrimaryHDU(data=cropped_data, header=header)
    # Save the cropped FITS file
    output_fits = ('/Users/claraaldegundemanteca/Desktop/HerschelField/Code/cutout_%s.fits'% str(ID))
    hdu.writeto(output_fits, overwrite=True)
    
    #Plot image to check we're seeing the right source
    if plot_image == True: 
        fig_source = aplpy.FITSFigure(output_fits)
        fig_source.show_colorscale(vmin = 0.1, vmax = 20, stretch = 'log', cmap='gist_heat')

    #Image to find Sersic index
    hdulist = fits.open(output_fits)
    image =  hdulist[0].data
    
    #Normalise image
    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val)    
    
    # Load PSF
    PSF = fits.getdata('/Users/claraaldegundemanteca/Desktop/HerschelField/Code/JWST_PSF.fits')
    PSF = PSF / PSF.sum() #normalise it

    #Initial guesses for the fit
    amplitude= A0
    r_eff= r_eff0
    n= n0
    x_0=x
    y_0=y
    ellip= ellip0
    theta= theta0

    center_slack = 20
    
    #Create model
    sersic_model = models.Sersic2D(
            amplitude=amplitude,
            r_eff=r_eff,
            n=n,
            x_0=x_0,
            y_0=y_0,
            ellip=ellip,
            theta=theta,
            bounds = pf.get_default_sersic_bounds({
                'x_0': (x_0 - center_slack/2, x_0 + center_slack/2),
                'y_0': (y_0 - center_slack/2, y_0 + center_slack/2),
            }))
            
            
    #Model for PSF
    psf_sersic_model = pf.PSFConvolvedModel2D(sersic_model, psf=PSF, oversample=4, psf_oversample=1)

    #Perform the fit
    fitting_weights = None
    fitted_model, fit_info = pf.fit_model(
        image, psf_sersic_model,
        weights=fitting_weights,
        calc_uncertainties=True,
        maxiter=10000,
        epsilon=1.4901161193847656e-08,
        acc=1e-09)
    
    fitted_model.cov_matrix
    param_stds = fitted_model.stds
    print(param_stds)
    #Plot fit 
    pf.plot_fit(fitted_model, image.data, figsize=[3*8, 3])
    #Produce list of params and list of errors
    plt.savefig('/Users/claraaldegundemanteca/Desktop/HerschelField/Code/Sérsic Models/Sérsic_model_%s.png'% (str(ID)), dpi = 300)
        
    list_params = [fitted_model.amplitude.value, fitted_model.r_eff.value, fitted_model.n.value, fitted_model.ellip.value, fitted_model.theta.value]
    list_std = param_stds.stds
    print(list_std)
    
    return list_params, list_std[:3] + list_std[5:7]  #without including x_0 and y_0

params = sersic_model_HST(ID=2007, radius = 100, input_fits = input_fits, A0 = 0.1, r_eff0 = 20, n0 = 4, x0 = 110, y0 = 130, ellip0 = 0.7, theta0 = 0.5 )
