#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:26:50 2023

@author: claraaldegundemanteca
"""

import numpy as np
from astropy.io import fits 
import matplotlib.pyplot as plt
from numpy import unravel_index
from Photometry_ell import *
#%%
#import data and plot histrogram 
hdulist = fits.open("A1_mosaic.fits")
#hdulist_clean = fits.open("A1_mosaic_no_artifs.fits")
hdulist_clean = fits.open("mask_updated.fits")

headers = hdulist[0].header
data_nd = hdulist[0].data
data_1d = np.array(data_nd)
data_1d = data_1d.flatten()

headers_clean = hdulist_clean[0].header
data_nd_clean = hdulist_clean[0].data
data_1d_clean = np.array(data_nd_clean)
data_1d_clean = data_1d_clean.flatten()

#%% plot whole image

y_vals = np.arange(0,4611,1)
x_vals = np.arange(0,2570,1)

x, y = np.meshgrid(x_vals, y_vals)
fig = plt.figure()
ax1 = plt.contourf(x, y, data_nd_clean)#increase contrast
cbar = fig.colorbar(ax1, cmap='cividis')
plt.show()
                      
#%% plot slice

import numpy as np
from astropy.io import fits 
import matplotlib.pyplot as plt
from numpy import unravel_index
from Photometry_ell import *

#import data and plot histrogram 
hdulist = fits.open("A1_mosaic.fits")
hdulist_clean = fits.open("mask_updated.fits")

headers = hdulist[0].header
data_nd = hdulist[0].data
data_1d = np.array(data_nd)
data_1d = data_1d.flatten()

headers_clean = hdulist_clean[0].header
data_nd_clean = hdulist_clean[0].data
data_1d_clean = np.array(data_nd_clean)
data_1d_clean = data_1d_clean.flatten()
#load image
fullimage_clean = data (data_nd_clean)

 

#plot slice
fullimage_clean.slicing(500, 500, 2, 0, plot_contour=True)


#%% counting sources

import numpy as np
from astropy.io import fits 
import matplotlib.pyplot as plt
from numpy import unravel_index
from Photometry_ell import *

#import data and plot histrogram 
hdulist = fits.open("A1_mosaic.fits")
hdulist_clean = fits.open("mask_updated.fits")

headers = hdulist[0].header
data_nd = hdulist[0].data
data_1d = np.array(data_nd)
data_1d = data_1d.flatten()

headers_clean = hdulist_clean[0].header
data_nd_clean = hdulist_clean[0].data
data_1d_clean = np.array(data_nd_clean)
data_1d_clean = data_1d_clean.flatten()
#load image
fullimage_clean = data (data_nd_clean)



magnitudes2=fullimage_clean.identify(500, 500, 2, 0)[0]

#%% total magnitudes


import numpy as np
from astropy.io import fits 
import matplotlib.pyplot as plt
from numpy import unravel_index
from Photometry_ell import *

#import data and plot histrogram 
hdulist = fits.open("A1_mosaic.fits")
hdulist_clean = fits.open("mask_updated.fits")
#hdulist_clean = fits.open("A1_mosaic_no_artifs.fits")


headers = hdulist[0].header
data_nd = hdulist[0].data
data_1d = np.array(data_nd)
data_1d = data_1d.flatten()

headers_clean = hdulist_clean[0].header
data_nd_clean = hdulist_clean[0].data
data_1d_clean = np.array(data_nd_clean)
data_1d_clean = data_1d_clean.flatten()
#load image
fullimage_clean = data (data_nd_clean)


x_pixels = 500
y_pixels = 500
number_x_boxes = fullimage_clean.slicing(x_pixels, y_pixels, 0, 0, plot_contour=False)[1]
number_y_boxes = fullimage_clean.slicing(x_pixels, y_pixels, 0, 0, plot_contour=False)[2]
       

# magnitudes_fixed=fullimage_clean.magnitude_distribution(x_pixels, y_pixels, number_x_boxes,number_y_boxes, fixed= True)[0]
#magnitudes_varying=fullimage_clean.magnitude_distribution(x_pixels, y_pixels, number_x_boxes,number_y_boxes, fixed = False)[0]
magnitudes_varying_ell=fullimage_clean.magnitude_distribution(x_pixels, y_pixels, number_x_boxes,number_y_boxes)[0]
#magnitudes_varying_ell=fullimage_clean.magnitude_distribution(x_pixels, y_pixels,2,3)[0]
#%%
# results for varying elliptical magnitudes. plots hist and log plot

from scipy.optimize import curve_fit

np.save('magnitudes_varying_elliptical.npy', magnitudes_varying_ell)
#%%
magnitudes_varying_list = []
for i in range(0, len(magnitudes_varying_ell)):
    for j in range (0, len(magnitudes_varying_ell[i])):
        print(magnitudes_varying_ell[i][j])
        magnitudes_varying_list.append(magnitudes_varying_ell[i][j])
#%%
def expected_trend_log (m, c):
    return 0.6*m + c

def no_square_degrees (x_pixels, y_pixels):
    x_degrees = 7.1666e-5 * x_pixels
    y_degrees = 7.1666e-5 * y_pixels
    return x_degrees * y_degrees 
no_square_degrees = no_square_degrees (4611, 2570)

below_array = np.arange(5,20,0.5)

counts_per_square_degree = []
for k in below_array:
    counts= len([i for i in magnitudes_varying_list if i < k])
    counts_per_square_degree.append(np.log(counts*0.006))
    

def gaussian(x, mu, sig, A):
    return A*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

 
counts, bins, _= plt.hist(magnitudes_varying_list, 20)
bin_center = bins[:-1] + np.diff(bins) / 2

mu_g = np.mean(magnitudes_varying_list)
sig_g = np.std(magnitudes_varying_list)
A_g =400
popt, pcov = curve_fit(gaussian, bin_center, counts, p0=[mu_g,sig_g,A_g])


plt.figure()
plt.hist(magnitudes_varying_list, color='k', alpha=0.5)
# plt.plot(below_array,gaussian(below_array,popt[0],popt[1],popt[2]), color='red')
plt.xlabel('Magnitude')
plt.ylabel('No. of sources')
plt.title('Using varying elliptical aperture')
plt.grid(alpha=0.5)
plt.savefig('Varying_elliptical_aperture_hist.jpeg', dpi=300)



plt.figure()
plt.plot(below_array, counts_per_square_degree, '^', markersize=4,color='k')
fit_varying,cov_varying=np.polyfit(below_array[8:-8], counts_per_square_degree[8:-8],1, w=None, cov=True)
pfit=np.poly1d(fit_varying)
plt.plot(below_array[8:-8], pfit(below_array[8:-8]), color= 'red', linewidth = 0.7)
plt.grid(alpha=0.5)
plt.ylabel('log N/$deg^2$')
plt.xlabel('mag')
plt.title('Using varying elliptical aperture')
plt.savefig('Varying_elliptical_aperture_logplot.jpeg', dpi=300)


