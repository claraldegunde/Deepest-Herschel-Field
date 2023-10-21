#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:15:31 2023

@author: claraaldegundemanteca
"""

import numpy as np
from astropy.io import fits 
import matplotlib.pyplot as plt
from numpy import unravel_index
from scipy.optimize import curve_fit

#import data
hdulist = fits.open("A1_mosaic.fits")

headers = hdulist[0].header
data_nd = hdulist[0].data

def gaussian( x, mu, sig,A):
    return A*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


class data: 
    def __init__(self, data2d):
        self._data2d = data2d
        y_vals = np.arange(0,len(self._data2d[:,1]),1) #size of y
        x_vals = np.arange(0,len(self._data2d[0,:]),1) #size of x
        self._x, self._y = np.meshgrid(x_vals, y_vals) # meshgrid fro plots
        self._source_no = 0 

    def contour_plot (self):
        fig = plt.figure()
        ax1 = plt.contourf(self._x, self._y, data_nd)
        cbar = fig.colorbar(ax1)
        plt.show()

    def slicing (self, x_pixels, y_pixels, n, m, plot_contour = False, plot_surface = False): # n is x coordinate of box, m is y coordinate
        '''
        Giveng dimensions of slice and n, m coordinates of the box we want to access, 
        return slice data and plots (if true). 
        Also returns number of x and y boxes to check sensible amount of them
        '''
        data_no_edges = self._data2d

        no_of_x_sections = int(len(data_no_edges[0,:])/x_pixels)+ 1 #add one because of the one left
        no_of_y_sections = int(len(data_no_edges[:,0])/y_pixels) + 1 # number of boxes for x and y 
        # print(no_of_x_sections, no_of_y_sections)
        
        #Will have 1 that is not the wanted dimension 
        pixels_left_x = len(data_no_edges[0])-no_of_x_sections*x_pixels #add them to the last one
        pixels_left_y = len( data_no_edges[:,1])-no_of_y_sections*y_pixels #add them to the last one
        # print(pixels_left_x, pixels_left_y)
        #Sliced data
        #Add leftover pixels if at the limit
        if m == (no_of_y_sections-1)  and  n == (no_of_x_sections -1 ): 
            slice_n_m = self._data2d[m*y_pixels:(m+1)*y_pixels + pixels_left_y, n*x_pixels:(n+1)*x_pixels + pixels_left_x]
            # y_vals = np.linspace(m*y_pixels,(m+1)*y_pixels+pixels_left_y,len(slice_n_m[:,0]))
            # x_vals = np.linspace(n*x_pixels,(n+1)*x_pixels+pixels_left_x,len(slice_n_m[0,:]))

        if m == (no_of_y_sections -1)  and  n != (no_of_x_sections -1 ): 
           slice_n_m = self._data2d[m*y_pixels:(m+1)*y_pixels + pixels_left_y, n*x_pixels:(n+1)*x_pixels]
        if n == (no_of_x_sections-1) and  m != (no_of_y_sections -1 ):
           slice_n_m = self._data2d[m*y_pixels:(m+1)*y_pixels, n*x_pixels:(n+1)*x_pixels+ pixels_left_x]

        if m != (no_of_y_sections-1)  and  n != (no_of_x_sections -1 ): 
            slice_n_m = self._data2d[m*y_pixels:(m+1)*y_pixels , n*x_pixels:(n+1)*x_pixels ]

        np.save(f'Slice_{n}_{m}.npy', slice_n_m)
       
        if plot_contour == True: 
            if m == (no_of_y_sections -1 ) and n == (no_of_x_sections -1 ): 
                y_vals = np.arange(m*y_pixels,(m+1)*y_pixels+pixels_left_y,1)
                x_vals = np.arange(n*x_pixels,(n+1)*x_pixels+pixels_left_x,1) #shifted so same coordinates as the image
            if m == (no_of_y_sections -1 ) and  n != (no_of_x_sections -1 ):
                y_vals = np.arange(m*y_pixels,(m+1)*y_pixels+pixels_left_y,1)
                x_vals = np.arange(n*x_pixels,(n+1)*x_pixels,1) #shifted so same coordinates as the image
            if m != (no_of_y_sections -1 ) and  n == (no_of_x_sections -1 ):
                y_vals = np.arange(m*y_pixels,(m+1)*y_pixels,1)
                x_vals = np.arange(n*x_pixels,(n+1)*x_pixels+pixels_left_x,1) 
            if m != (no_of_y_sections -1 ) and  n != (no_of_x_sections -1 ):
                y_vals = np.arange(m*y_pixels,(m+1)*y_pixels,1)
                x_vals = np.arange(n*x_pixels,(n+1)*x_pixels,1) 
            x, y = np.meshgrid(x_vals, y_vals)

            z = slice_n_m
            # print(len(z[0]),len(z[:,0]))

            fig = plt.figure()
            ax1 = plt.contourf(x, y, z)
            cbar = fig.colorbar(ax1)
            plt.title(f'Slice_{n}_{m}')
            plt.show()
        if plot_surface == True:
            fig= plt.figure()
            ax2 = plt.axes(projection = '3d')
            ax2.plot_surface (x,y,z, cmap= 'viridis')
            plt.title(f'Slice_{n}_{m}')
            plt.show()
        return slice_n_m, no_of_x_sections, no_of_y_sections
    
    def find_maximum (self, x_pixels, y_pixels, n, m, plot_contour = False, plot_surface = False):
        '''
        Parameters
        ----------
        x_pixels : no pixels x direction for slice
        y_pixels : no pixels y direction for slice
        n : x coordinate of slice
        m : y coordinate of slice
        
        Returns
        -------
        x_y_coordinates : Coordinates of maximum
         n_m_slice[x_y_coordinates] : value of maximum 

        '''
        n_m_slice = self.slicing(x_pixels, y_pixels, n, m, plot_contour = False, plot_surface = False)[0]
        max_index = n_m_slice.argmax()
        x_y_coordinates = unravel_index(max_index, n_m_slice.shape)
        return [x_y_coordinates, n_m_slice[x_y_coordinates]]

    def circular_aperture(self, c, x_pixels, y_pixels, n, m, r= 12,  remove = False, plot = False):
        
        '''
        Parameters 
        ----------
        h = height of the slice
        w = width  of the slice
        c = centre of the aperture 
        r = radius of the aperture
        
        Returns
        -------
        region = array with shape of slice, boolean with true as the aperture region
        
        '''
        if m == 10 and n == 6: 
            h, w = y_pixels + 111, x_pixels + 70  
        if m == 10 and  n != 6:
            h, w = y_pixels + 111, x_pixels  
        if m != 10 and  n == 6:
            h, w = y_pixels, x_pixels + 70  
        if m != 10 and  n !=6:
            h, w = y_pixels, x_pixels 
            
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - c[1])**2 + (Y-c[0])**2) ###change 
        region = dist_from_center <= r
        
            
        if plot == True:
            visual = np.copy(self.slicing(x_pixels, y_pixels, n, m, plot_contour = False, plot_surface = False)[0])
            visual[region]  = 0
            
            y_vals = np.arange(m*y_pixels,(m+1)*y_pixels,1)
            x_vals = np.arange(n*y_pixels,(n+1)*x_pixels,1) #shifted so same coordinates as the image
            x, y = np.meshgrid(x_vals, y_vals)
            z = visual
            fig = plt.figure()
            ax1 = plt.contourf(x, y, z)
            cbar = fig.colorbar(ax1)
            plt.title(f'Slice_{n}_{m}')
            plt.show()             
        return region 
    

    def aperture_radius(self, c, x_pixels, y_pixels, n, m):
        
        h, w = y_pixels, x_pixels
        x, y = c[0], c[1]
        
        data_nd = np.copy(self.slicing(x_pixels, y_pixels, n, m, plot_contour = False, plot_surface = False)[0])
        sub_set_left =  np.flip(data_nd[x][0:y])
        
        if len([ n for n,i in enumerate(sub_set_left) if i<3500]) == 0:
            left_rad = 0
        else:
            left_rad =  [ n for n,i in enumerate(sub_set_left) if i<3500][0] #3600 is background mean 

        
        # left_rad = [ n for n, i in enumerate(sub_set_left) if i<3500][0]  # change to global mean 
        
        sub_set_right = data_nd[x][y:]
        if len([ n for n,i in enumerate(sub_set_right) if i<3500]) == 0:
            right_rad = 0
        else:
            right_rad =  [ n for n,i in enumerate(sub_set_right) if i<3500][0] #3600 is background mean 

        rad = np.max([left_rad,right_rad])
        return rad
        
         
    def annulus_region(self, x_pixels, y_pixels, n, m,  c, r1, r2, plot = False):     
        
        '''
        Parameters 
        ----------
        h = height of the slice
        w = width  of the slice
        c = centre of the aperture 
        r1 = inner radius of the annulus 
        r2 = outter radius of the annulus 
        
        
        Returns 
        ------
        mu = mean of the background values
        sig = standard deviation of the background values
        region  = array with shape of slice, boolean with true as the annulus region
        '''
        if m == 10 and n == 6: 
            h, w = y_pixels + 111, x_pixels + 70  
        if m == 10 and  n != 6:
            h, w = y_pixels + 111, x_pixels  
        if m != 10 and  n == 6:
            h, w = y_pixels, x_pixels + 70  
        if m != 10 and  n !=6:
            h, w = y_pixels, x_pixels 


        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - c[1])**2 + (Y-c[0])**2)
        a = dist_from_center <= r2
        b = dist_from_center >= r1
        region = np.logical_and(a,b)
        
        if plot == True:
            
            visual = np.copy(self.slicing(x_pixels, y_pixels, n, m, plot_contour = False, plot_surface = False)[0])#.deepcopy()
            visual[region]  = -1000
            y_vals = np.arange(m*y_pixels,(m+1)*y_pixels,1)
            x_vals = np.arange(n*y_pixels,(n+1)*x_pixels,1)
            x, y = np.meshgrid(x_vals, y_vals)
            z = visual
            fig = plt.figure()
            ax1 = plt.contourf(x, y, z)
            cbar = fig.colorbar(ax1)
            plt.title(f'Zoomed in plot visualise')
            plt.xlim(n*y_pixels,(n+1)*x_pixels)
            plt.ylim(m*y_pixels,(m+1)*y_pixels)
            plt.show()             
        
        
        # work with slices not with whole data 
        
        data_filtered = np.copy(self.slicing(x_pixels, y_pixels, n, m, plot_contour = False, plot_surface = False)[0])
        data_filtered = data_filtered[region].flatten()
        
        data_filtered = data_filtered[data_filtered != 0]
        data_filtered = data_filtered[data_filtered != 1]
        data_filtered = data_filtered[data_filtered < 4000]

        
        
        #fit to histogram 
        # plt.figure()
        # counts, bins, _ = plt.hist(data_filtered)
        # bin_center = bins[:-1] + np.diff(bins) / 2
        
        mu_g = np.mean(np.copy(self.slicing(x_pixels, y_pixels, n, m, plot_contour = False, plot_surface = False)[0])[region]) 
        sig_g = np.std(np.copy(self.slicing(x_pixels, y_pixels, n, m, plot_contour = False, plot_surface = False)[0])[region])
        # A_g = max(counts)
        
        # popt, pcov = curve_fit(gaussian, bin_center, counts, p0=[mu_g,sig_g,A_g])
        # mu, sig = popt[0], popt[1]
        # x_gau = np.linspace(3200,3600,100)
        # y_gau = gaussian(x_gau, popt[0], popt[1], popt[2])
        # plt.plot(x_gau, y_gau)
        
        return mu_g, sig_g, region
    
    
    def remove_source(self, x_pixels, y_pixels, n, m, region, bg_mu):
        
        slice_= self.slicing(x_pixels, y_pixels, n, m, plot_contour = False, plot_surface = False)[0]
        
        pixel_intensity = np.copy(slice_[region]).flatten() 
        pixel_intensity = pixel_intensity - bg_mu
        total_intensity = sum(pixel_intensity)
        
        slice_[region]= 1 # source analysed  
        self._source_no += 1 
        
        
        y_vals = np.arange(m*y_pixels,(m+1)*y_pixels,1)
        x_vals = np.arange(n*y_pixels,(n+1)*x_pixels,1) #shifted so same coordinates as the image
        x, y = np.meshgrid(x_vals, y_vals)
        z = slice_
        # fig = plt.figure()
        # ax1 = plt.contourf(x, y, z)
        # cbar = fig.colorbar(ax1)
        # plt.title(f'Slice_{n}_{m}')
        # plt.show()  
        return total_intensity
    
    
    def identify (self, x_pixels, y_pixels, n, m, fixed= True):
        """
        Returns list of intensities, coordinates and number of sources found 
        
        Stop when maximum is less than 3600

        """
        
        magnitudes = []
        magnitude_errors = []
        centres=[] #location
        for i in range(0,100): #give 1000, large number
            #find centre
            centre = self.find_maximum(x_pixels, y_pixels, n, m)[0]
            if self.find_maximum(x_pixels, y_pixels, n, m)[1] < 3550: 
                break
            else: 
                if fixed == True:#for fixed aperture 
                    radius = 12
                if fixed == False: 
                    radius = self.aperture_radius(centre, x_pixels, y_pixels, n, m)

                #find region 
                region = self.circular_aperture(centre,x_pixels, y_pixels,  n, m, r =  radius , remove = True, plot = False)
                
                #find background
                mu, sig, _ = self.annulus_region(x_pixels, y_pixels,n, m,centre, r1 = radius, r2=radius+5, plot = False)
                
                #remove source and find intensity 
                intensity = self.remove_source(x_pixels,  y_pixels, n, m, region = region, bg_mu = mu)
                if intensity > 0:
                    magnitude = 25.3-2.5*np.log10(intensity)
                    err_counts = 0 #change when how to define error has been decided 
                    err_magnitude = 0.02 - 2.5*err_counts/(np.log(10)*intensity)#0.02 is the error on Pinst given by fits
                    magnitudes.append(magnitude)
                    magnitude_errors.append(err_magnitude)
        # centres.append(centre) 
        magnitudes = np.array(magnitudes)
        centres = np.array(centres)

        return magnitudes, centres, len(centres)


    def magnitude_distribution (self, x_pixels, y_pixels, last_x_slice, last_y_slice, fixed=True): # given limits of x and y boxes
        magnitudes = []
        centres = []     
        for m in range (0, last_y_slice):
            for n in range (0, last_x_slice ):
                # if n == last_x_slice and m == last_y_slice :
                #     magnitudes.append(self.identify(x_pixels+70, y_pixels+111, n, m)[0])
                #     centres.append(self.identify(x_pixels+70, y_pixels+111, n, m)[1])
                # if n == last_x_slice and m != last_y_slice :
                #     magnitudes.append(self.identify(x_pixels+70, y_pixels, n, m)[0])
                #     centres.append(self.identify(x_pixels+70, y_pixels, n, m)[1])
                # if n != last_x_slice and m == last_y_slice :
                #     magnitudes.append(self.identify(x_pixels, y_pixels+111, n, m)[0])
                #     centres.append(self.identify(x_pixels, y_pixels+111, n, m)[1])
                # if n != last_x_slice and m != last_y_slice :
                magnitudes.append(self.identify(x_pixels, y_pixels, n, m, fixed)[0])
                centres.append(self.identify(x_pixels, y_pixels, n, m, fixed)[1])                  
                print(n, m)
        magnitudes = np.array(magnitudes)
        centres = np.array(centres)
                # print(centres)
        return magnitudes, centres, len(centres)

#to get the radius for the varying aperture you requrie knowldge of background mean
#but you get that once you have the radius 
    
    

            
            