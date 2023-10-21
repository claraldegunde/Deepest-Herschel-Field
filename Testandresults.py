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
from Photometry import *
#%%
#import data and plot histrogram 
hdulist = fits.open("A1_mosaic.fits")
hdulist_clean = fits.open("A1_mosaic_no_artifs.fits")

headers = hdulist[0].header
data_nd = hdulist[0].data
data_1d = np.array(data_nd)
data_1d = data_1d.flatten()

headers_clean = hdulist_clean[0].header
data_nd_clean = hdulist_clean[0].data
data_1d_clean = np.array(data_nd_clean)
data_1d_clean = data_1d_clean.flatten()

#%%
##fit a gaussian to hist
#def gaussian(x, mu, sig, A):
#    return A*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
#
##full data
#mu_g = np.mean(data_1d)
#sig_g = np.std(data_1d)
#A_g = max(counts)
#popt, pcov = curve_fit(gaussian, bin_center, counts, p0=[mu_g,sig_g,A_g])
#
#plt.figure()
#counts, bins, _ = plt.hist(data_1d,20000)
#bin_center = bins[:-1] + np.diff(bins) / 2
#plt.xlim(3300,3500)
#plt.xlabel('Counts per pixel')
#plt.ylabel('Frequency')
#plt.show()
#x = np.linspace(3000,4000,4000)
#y = gaussian(x,popt[0],popt[1],popt[2])
#plt.plot(x,y)
#plt.show()
#
#mu_g = np.mean(data_1d_clean)
#sig_g = np.std(data_1d_clean)
#A_g = max(counts)
#popt, pcov = curve_fit(gaussian, bin_center, counts, p0=[mu_g,sig_g,A_g])
#
## plt.figure()
#counts, bins, _ = plt.hist(data_1d_clean,20000, label='No artifacts')
#bin_center = bins[:-1] + np.diff(bins) / 2
#plt.xlim(3300,3500)
#plt.xlabel('Counts per pixel')
#plt.ylabel('Frequency')
#plt.show()
#x = np.linspace(3000,4000,4000)
#y = gaussian(x,popt[0],popt[1],popt[2])
#plt.plot(x,y)
#plt.legend()
#plt.show() 
##masks
##%%
##looking at real light sources
#mu = popt[0]
#sig = popt[1]
#plt.figure()
#plt.hist(data_1d,10000)
#plt.xlim(mu+sig*5,50000)
#plt.ylim(0,300)
#plt.savefig('Zoomed hist.png',dpi=600)
#plt.xlabel('Counts per pixel')
#plt.ylabel('Frequency')

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
from Photometry import *

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
fullimage_clean.slicing(500, 500, 5, 0, plot_contour=True)


#%% counting sources

import numpy as np
from astropy.io import fits 
import matplotlib.pyplot as plt
from numpy import unravel_index
from Photometry import *

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



magnitudes2=fullimage_clean.identify(500, 500, 0, 9, fixed=False)

#%% checking annulus

import numpy as np
from astropy.io import fits 
import matplotlib.pyplot as plt
from numpy import unravel_index
from Photometry import *

#import data and plot histrogram 
hdulist = fits.open("A1_mosaic.fits")
hdulist_clean = fits.open("A1_mosaic_no_artifs.fits")

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
c= fullimage_clean.find_maximum (500, 500, 2, 0, plot_contour = False, plot_surface = False)[0]
r1=fullimage_clean.aperture_radius(c, 500, 500, 2, 0)

fullimage_clean.annulus_region(500, 500, 2, 0, c, r1, r2=2*r1, plot = True)

#%% total magnitudes


import numpy as np
from astropy.io import fits 
import matplotlib.pyplot as plt
from numpy import unravel_index
from Photometry_2 import *

#import data and plot histrogram 
hdulist = fits.open("A1_mosaic.fits")
# hdulist_clean = fits.open("mask_updated.fits")
hdulist_clean = fits.open("A1_mosaic_no_artifs.fits")


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
      
# Fixed 12 pixels 
# magnitudes_fixed_12=fullimage_clean.magnitude_distribution(x_pixels, y_pixels, number_x_boxes, number_y_boxes,12 , fixed= True)
# np.save('magnitudes_fixed_12.npy', magnitudes_fixed_12[0])
# magnitudes_fixed_12=np.load('magnitudes_fixed_12.npy', allow_pickle=True)


# Varying aperture 
# magnitudes_varying=fullimage_clean.magnitude_distribution(x_pixels, y_pixels, number_x_boxes,number_y_boxes, radius_fixed=0, fixed = False)
# np.save('magnitudes_varying.npy', magnitudes_varying[0])
# magnitudes_varying=np.load('magnitudes_varying.npy', allow_pickle=True)


#1 sigma
# magnitudes_1sigma=np.load('magnitudes_1sigma.npy', allow_pickle=True)

#1.3 sigma
magnitudes_13sigma=np.load('magnitudes_13sigma.npy', allow_pickle=True)


       
#%% results for varying circular magnitudes. plots hist and log plot

from scipy.optimize import curve_fit


magnitudes_varying_list = []
for i in range(0, len(magnitudes_varying)):
    for j in range (0, len(magnitudes_varying[i])):
        magnitudes_varying_list.append(magnitudes_varying[i][j])

def expected_trend_log (m, c):
    return 0.6*m + c

def no_square_degrees (x_pixels, y_pixels):
    x_degrees = 7.1666e-5 * x_pixels
    y_degrees = 7.1666e-5 * y_pixels
    return x_degrees * y_degrees 
no_square_degrees = no_square_degrees (4611-2*119, 2570-2*117)

min_m=np.min(magnitudes_varying_list)
below_array_varying = np.arange(int(min_m)+0.7,25,0.5)

counts_per_square_degree_varying = []
err_counts_per_square_degree_varying = []
err_mag = []
for k in below_array_varying:
    counts = len([i for i in magnitudes_varying_list if i < k])
    err=np.sqrt(counts)/(counts*np.log(10))
    counts_per_square_degree_varying.append(np.log10(counts/no_square_degrees))
    err_counts_per_square_degree_varying.append(err)
    err_mag.append(0.02)
        
        

    
def gaussian(x, mu, sig, A):
    return A*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

plt.figure()
counts, bins, _= plt.hist(magnitudes_varying_list, 20)
bin_center = bins[:-1] + np.diff(bins) / 2

mu_g = np.mean(magnitudes_varying_list)
sig_g = np.std(magnitudes_varying_list)
A_g = max(10*counts)
popt, pcov = curve_fit(gaussian, bin_center, counts, p0=[mu_g,sig_g,A_g])


# plt.figure()
# plt.hist(magnitudes_varying_list, color='k', alpha=0.5)
# # plt.plot(below_array_varying,gaussian(below_array_varying,popt[0],popt[1],popt[2]), color='red')
# plt.xlabel('Magnitude')
# plt.ylabel('No. of sources')
# plt.title('Varying circular aperture')
# plt.grid(alpha=0.5)
# plt.savefig('varying_aperture_hist.jpeg', dpi=300)



plt.figure()
plt.errorbar(below_array_varying, counts_per_square_degree_varying, xerr = err_mag, fmt='^', markersize=3, capsize = 2,color='k')
fit_varying,cov_varying=np.polyfit(below_array_varying[6:-14], counts_per_square_degree_varying[6:-14],1, w=None, cov=True)
pfit=np.poly1d(fit_varying)
plt.plot(below_array_varying[6:-14], pfit(below_array_varying[6:-14]), color= 'red', linewidth = 0.7)
plt.grid(alpha=0.5)
plt.ylabel('log N/$deg^2$')
plt.xlabel('mag')
plt.title('Varying circular aperture')
plt.savefig('varying_aperture_logplot.jpeg', dpi=300)


#%% results for fixed circular magnitudes. plots hist and log plot

min_m=np.min(magnitudes_fixed_list)
below_array_fixed = np.arange(int(min_m)+0.5,25,0.5)

from scipy.optimize import curve_fit
magnitudes_fixed=magnitudes_fixed_12
magnitudes_fixed_list = []
for i in range(0, len(magnitudes_fixed)-1):
    for j in range (0, len(magnitudes_fixed[i])-1):
        magnitudes_fixed_list.append(magnitudes_fixed[i][j])


counts_per_square_degree_fixed = []
err_counts_per_square_degree_fixed = []
err_mag = []
for k in below_array_fixed:
    counts = len([i for i in magnitudes_fixed_list if i < k])
    err=np.sqrt(counts)/(counts*np.log(10))
    counts_per_square_degree_fixed.append(np.log10(counts/no_square_degrees))
    err_counts_per_square_degree_fixed.append(err)
    err_mag.append(0.02)


plt.figure()
counts, bins, _= plt.hist(magnitudes_fixed_list, 20)
bin_center = bins[:-1] + np.diff(bins) / 2

mu_g = np.mean(magnitudes_fixed_list)
sig_g = np.std(magnitudes_fixed_list)
A_g = max(10*counts)
popt, pcov = curve_fit(gaussian, bin_center, counts, p0=[mu_g,sig_g,A_g])


# plt.figure()
# plt.hist(magnitudes_fixed_list, color='k', alpha=0.5)
# # plt.plot(below_array_fixed,gaussian(below_array_fixed,popt[0],popt[1],popt[2]), color='red')
# plt.xlabel('Magnitude')
# plt.ylabel('No. of sources')
# plt.title('fixed circular aperture')
# plt.grid(alpha=0.5)
# plt.savefig('fixed_aperture_hist.jpeg', dpi=300)


plt.figure()
plt.errorbar(below_array_fixed, counts_per_square_degree_fixed, xerr = err_mag, fmt='^', markersize=3, capsize = 2,color='k')
# fit_fixed,cov_fixed=np.polyfit(below_array_fixed[8:-7], counts_per_square_degree_fixed[8:-7],1, w=None, cov=True)
pfit=np.poly1d(fit_fixed)
# plt.plot(below_array_fixed[8:-7], pfit(below_array_fixed[8:-7]), color= 'red', linewidth = 0.7)
plt.grid(alpha=0.5)
plt.ylabel('log N/$deg^2$')
plt.xlabel('mag')
plt.title('fixed circular aperture')
plt.savefig('whole_fixed_12.jpeg', dpi=300)

#%% plot both fixed and varyin in same plot 
plt.figure()
plt.errorbar(below_array_varying, counts_per_square_degree_varying, xerr = err_mag, fmt='o', markersize=4, capsize = 2,color='black', label='Varying circular')
fit_varying,cov_varying=np.polyfit(below_array_varying[8:-5], counts_per_square_degree_varying[8:-5],1, w=None, cov=True)
pfit=np.poly1d(fit_varying)
plt.plot(below_array_varying[8:-5], pfit(below_array_varying[8:-5]), color= 'black', linewidth = 0.7)

plt.errorbar(below_array_fixed, counts_per_square_degree_fixed, xerr = err_mag, fmt='x', markersize=6, capsize = 2,color='k', label='Fixed circular (12 pixels)')
fit_fixed,cov_fixed=np.polyfit(below_array_fixed[8:-7], counts_per_square_degree_fixed[8:-7],1, w=None, cov=True)
pfit=np.poly1d(fit_fixed)
plt.plot(below_array_fixed[8:-7], pfit(below_array_fixed[8:-7]), color= 'k', linewidth = 0.7)
plt.grid(alpha=0.5)
plt.ylabel('log N/$deg^2$')
plt.xlabel('mag')
plt.title('Comparing circular apertures')
plt.legend()
plt.savefig('Fixed_and_varying_logplot_pixels_12.jpeg', dpi=300)



#%% elliptical 1 sigma 
min_m=np.min(magnitudes_1sigma_list)
below_array_1sigma = np.arange(int(min_m)+0.5,25,0.5)

from scipy.optimize import curve_fit


magnitudes_1sigma_list = []
for i in range(0, len(magnitudes_1sigma)):
    for j in range (0, len(magnitudes_1sigma[i])):
        magnitudes_1sigma_list.append(magnitudes_1sigma[i][j])



counts_per_square_degree_1sigma = []
err_counts_per_square_degree_1sigma = []
err_mag_1 = []
for k in below_array_1sigma:
    counts = len([i for i in magnitudes_1sigma_list if i < k])
    err=np.sqrt(counts)/(counts*np.log(10))
    counts_per_square_degree_1sigma.append(np.log10(counts/no_square_degrees))
    err_counts_per_square_degree_1sigma.append(err)
    err_mag_1.append(0.02)

    

plt.figure() 
counts, bins, _= plt.hist(magnitudes_1sigma_list, 20)
bin_center = bins[:-1] + np.diff(bins) / 2

mu_g = np.mean(magnitudes_1sigma_list)
sig_g = np.std(magnitudes_1sigma_list)
A_g = max(10*counts)
popt, pcov = curve_fit(gaussian, bin_center, counts, p0=[mu_g,sig_g,A_g])


# plt.figure()
# plt.hist(magnitudes_1sigma_list, color='k', alpha=0.5)
# # plt.plot(below_array_1sigma,gaussian(below_array_1sigma,popt[0],popt[1],popt[2]), color='red')
# plt.xlabel('Magnitude')
# plt.ylabel('No. of sources')
# plt.title('1sigma circular aperture')
# plt.grid(alpha=0.5)
# plt.savefig('1sigma_aperture_hist.jpeg', dpi=300)



plt.figure()
plt.errorbar(below_array_1sigma, counts_per_square_degree_1sigma, xerr = err_mag_1, fmt='^', markersize=3, capsize = 2,color='k')
fit_1sigma,cov_1sigma=np.polyfit(below_array_1sigma[8:-5], counts_per_square_degree_1sigma[8:-5],1, w=None, cov=True)
pfit=np.poly1d(fit_1sigma)
plt.plot(below_array_1sigma[8:-5], pfit(below_array_1sigma[8:-5]), color= 'red', linewidth = 0.7)
plt.grid(alpha=0.5)
plt.ylabel('log N/$deg^2$')
plt.xlabel('mag')
plt.title('1sigma aperture')
plt.savefig('1sigma_aperture_logplot.jpeg', dpi=300)


#%% 1.3 sigma 

magnitudes_13sigma_list = []
for i in range(0, len(magnitudes_13sigma)):
    for j in range (0, len(magnitudes_13sigma[i])):
        magnitudes_13sigma_list.append(magnitudes_13sigma[i][j])


min_m=np.min(magnitudes_13sigma_list)
below_array_13sigma = np.arange(int(min_m)+1,25,0.5)

counts_per_square_degree_13sigma = []
err_counts_per_square_degree_13sigma = []
err_mag_13 = []
for k in below_array_13sigma:
    counts = len([i for i in magnitudes_13sigma_list if i < k])
    err=np.sqrt(counts)/(counts*np.log(10))
    counts_per_square_degree_13sigma.append(np.log10(counts/no_square_degrees))
    err_counts_per_square_degree_13sigma.append(err)
    err_mag_13.append(0.02)



plt.figure() 
counts, bins, _= plt.hist(magnitudes_13sigma_list, 20)
bin_center = bins[:-1] + np.diff(bins) / 2

mu_g = np.mean(magnitudes_13sigma_list)
sig_g = np.std(magnitudes_13sigma_list)
A_g = max(10*counts)
popt, pcov = curve_fit(gaussian, bin_center, counts, p0=[mu_g,sig_g,A_g])


# plt.figure()
# plt.hist(magnitudes_13sigma_list, color='k', alpha=0.5)
# # plt.plot(below_array_13sigma,gaussian(below_array_13sigma,popt[0],popt[1],popt[2]), color='red')
# plt.xlabel('Magnitude')
# plt.ylabel('No. of sources')
# plt.title('13sigma circular aperture')
# plt.grid(alpha=0.5)
# plt.savefig('13sigma_aperture_hist.jpeg', dpi=300)


#%% plotting different sigmas together 
plt.figure()
# plt.subplot(1,2,1)
plt.errorbar(below_array_1sigma, counts_per_square_degree_1sigma, xerr = err_mag_1, fmt='o', markersize=4, capsize = 2,color='k', label='1 $\sigma$')
fit_1sigma,cov_1sigma=np.polyfit(below_array_1sigma[:-18], counts_per_square_degree_1sigma[:-18],1, w=None, cov=True)
pfit=np.poly1d(fit_1sigma)
plt.plot(below_array_1sigma[:-18], pfit(below_array_1sigma[:-18]), color= 'k', linewidth = 0.7)
plt.xlabel('mag')
plt.ylabel('log N/$deg^2$')
plt.title('1 $\sigma$ aperture')
plt.grid(alpha=0.5)

# plt.subplot(1,2,2)
plt.errorbar(below_array_13sigma, counts_per_square_degree_13sigma, xerr = err_mag_13, fmt='x', markersize=6, capsize = 2,color='r', label='1.3 $\sigma$')
fit_13sigma,cov_13sigma=np.polyfit(below_array_13sigma[:-18], counts_per_square_degree_13sigma[:-18],1, w=None, cov=True)
pfit=np.poly1d(fit_13sigma)
plt.plot(below_array_13sigma[:-18], pfit(below_array_13sigma[:-18]), color= 'r', linewidth = 0.7)
# plt.plot(below_array_1sigma[:-18], pfit(below_array_1sigma[:-18]), color= 'red', linewidth = 0.7)

# plt.grid(alpha=0.5)
plt.xlabel('mag')
# plt.legend()
plt.title('Comparing $\sigma$')
plt.legend()
plt.savefig('Compare_sigmas_logplot.jpeg', dpi=300)

