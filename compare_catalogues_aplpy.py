#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:54:30 2023

@author: claraaldegundemanteca
"""
import astropy 
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
import aplpy
import matplotlib.pyplot as plt
# documentation: https://buildmedia.readthedocs.org/media/pdf/aplpy/v0.9.11/aplpy.pdf

def from_deg_to_hmsdms (ra, dec): #in degrees, as in the catalogue
    '''
    Converts from ra and dec (catalogues) to hmsdms (ds9)
    '''
    coordinates = SkyCoord(ra, dec, frame=FK5, unit = 'deg') # in degrees
    return coordinates.to_string('hmsdms')

#%% CODE TO COMPARE FITS FROM DIFFERENT IMAGES

fig = plt.figure(figsize=(15,7))


#Centering
ra = 264.52176	 # directly in degrees from catalogue
dec = 69.0346


#Import fits
irac1 = aplpy.FITSFigure('IRAC/IRAC1.fits', figure = fig, subplot=(2,4,1))

#Greyscale
# gc.show_grayscale()

#Customise colorscale here
irac1.show_colorscale(cmap='twilight') 

#To check coordinates in plot
hmsdms = from_deg_to_hmsdms (ra, dec)

# x, y = gc.world2pixel(ra, dec)
irac1.recenter(ra, dec, radius = 0.01)

#Contours
irac1.show_contour('IRAC/IRAC1.fits', colors='white', filled= False)

#Customise axis 
irac1.axis_labels.hide_x()
irac1.tick_labels.set_font(size='small')


#Add title 
irac1.set_title('IRAC 1 (3.6 $\mu$m)')

#Add grid 
irac1.add_grid()
irac1.grid.set_color('white')
irac1.grid.set_alpha(0.5)

# Add markers
irac1.show_markers(ra, dec, edgecolor='yellow', marker='o', s=30, linestyle= '--', alpha=1)

#%%


#Import fits
irac2 = aplpy.FITSFigure('IRAC/IRAC2.fits', figure = fig, subplot=(2,4,2))

#Greyscale
# gc.show_grayscale()

#Customise colorscale here
irac2.show_colorscale(cmap='twilight') 


#To check coordinates in plot
hmsdms = from_deg_to_hmsdms (ra, dec)

# x, y = gc.world2pixel(ra, dec)
irac2.recenter(ra, dec, radius = 0.01)

#Contours
irac2.show_contour('IRAC/IRAC2.fits', colors='white', filled= False)

#Customise axis 
irac2.axis_labels.hide()
irac2.tick_labels.hide_y()
irac2.tick_labels.set_font(size='small')

#Add grid 
irac2.add_grid()
irac2.grid.set_color('white')
irac2.grid.set_alpha(0.5)

#Add title 
irac2.set_title('IRAC 2 (4.5 $\mu$m)')

# Add markers
irac2.show_markers(ra, dec, edgecolor='yellow', marker='o', s=30, linestyle= '--', alpha=1)

#%%
#Import fits
irac3 = aplpy.FITSFigure('IRAC/IRAC3.fits', figure = fig, subplot=(2,4,3))

#Greyscale
# gc.show_grayscale()

#Customise colorscale here
irac3.show_colorscale(cmap='twilight') 

#To check coordinates in plot
hmsdms = from_deg_to_hmsdms (ra, dec)

# x, y = gc.world2pixel(ra, dec)
irac3.recenter(ra, dec, radius = 0.01)

#Contours
irac3.show_contour('IRAC/IRAC3.fits', colors='white', filled= False)

#Customise axis 
irac3.axis_labels.hide()
irac3.tick_labels.hide_y()
irac3.tick_labels.set_font(size='small')

#Add grid 
irac3.add_grid()
irac3.grid.set_color('white')
irac3.grid.set_alpha(0.5)

#Add title 
irac3.set_title('IRAC 3 (5.8 $\mu$m)')

# Add markers
irac3.show_markers(ra, dec, edgecolor='yellow', marker='o',s=30, linestyle= '--', alpha=1)

#%%

#Import fits
irac4 = aplpy.FITSFigure('IRAC/IRAC4.fits', figure = fig, subplot=(2,4,4))

#Greyscale
# gc.show_grayscale()

#Customise colorscale here
irac4.show_colorscale(cmap='twilight') 


#To check coordinates in plot
hmsdms = from_deg_to_hmsdms (ra, dec)

# x, y = gc.world2pixel(ra, dec)
irac4.recenter(ra, dec, radius = 0.01)

#Contours
irac4.show_contour('IRAC/IRAC4.fits', colors='white', filled= False)

#Customise axis 
irac4.axis_labels.hide()
irac4.tick_labels.hide_y()
irac4.tick_labels.set_font(size='small')

#Add grid 
irac4.add_grid()
irac4.grid.set_color('white')
irac4.grid.set_alpha(0.5)

#Add title 
irac4.set_title('IRAC 4 (8.5 $\mu$m)')

# Add markers
irac4.show_markers(ra, dec, edgecolor='yellow', marker='o',s=30, linestyle= '--', alpha=1)


#%%
#Import fits
mips24 = aplpy.FITSFigure('MIPS/MIPS24.fits', figure = fig, subplot=(2,4,5))

#Greyscale
# gc.show_grayscale()

#Customise colorscale here
mips24.show_colorscale(cmap='twilight') 


# Customise ticks
mips24.tick_labels.set_font(size='small')


#To check coordinates in plot
hmsdms = from_deg_to_hmsdms (ra, dec)

# x, y = gc.world2pixel(ra, dec)
mips24.recenter(ra, dec, radius = 0.01)

#Contours
mips24.show_contour('MIPS/MIPS24.fits', colors='white', filled= False)

#Add grid 
mips24.add_grid()
mips24.grid.set_color('white')
mips24.grid.set_alpha(0.5)

#Add title 
mips24.set_title('MIPS 24 (24 $\mu$m)')

# Add markers
mips24.show_markers(ra, dec, edgecolor='yellow', marker='o',s=30, linestyle= '--', alpha=1)




#%% 

#Import fits
psw = aplpy.FITSFigure('SPIRE/PSW_masked.fits', figure = fig, subplot=(2,4,6))

#Greyscale
# gc.show_grayscale()

#Customise colorscale here
psw.show_colorscale(cmap='twilight') 


# Customise font
psw.tick_labels.set_font(size='small')


#To check coordinates in plot
hmsdms = from_deg_to_hmsdms (ra, dec)

# x, y = gc.world2pixel(ra, dec)
psw.recenter(ra, dec, radius = 0.01)

#Contours
psw.show_contour('SPIRE/PSW_masked.fits', colors='white', filled= False)

#Customise axis 
psw.axis_labels.hide_y()
psw.tick_labels.hide_y()
psw.tick_labels.set_font(size='small')

#Add grid 
psw.add_grid()
psw.grid.set_color('white')
psw.grid.set_alpha(0.5)

#Add title 
psw.set_title('PSW (250 $\mu$m)')

# Add markers
psw.show_markers(ra, dec, edgecolor='yellow', marker='o',s=30, linestyle= '--', alpha=1)


#%%


#Import fits
pmw = aplpy.FITSFigure('SPIRE/PMW_masked.fits', figure = fig, subplot=(2,4,7))

#Greyscale
# gc.show_grayscale()

#Customise colorscale here
pmw.show_colorscale(cmap='twilight') 


# x, y = gc.world2pixel(ra, dec)
pmw.recenter(ra, dec, radius = 0.01)

#Contours
pmw.show_contour('SPIRE/PMW_masked.fits', colors='white', filled= False)


# Customise ticks
pmw.tick_labels.set_font(size='small')
pmw.tick_labels.hide_y()
pmw.axis_labels.hide_y()

#Add grid 
pmw.add_grid()
pmw.grid.set_color('white')
pmw.grid.set_alpha(0.5)


#Add title 
pmw.set_title('PMW (350 $\mu$m)')

# Add markers
pmw.show_markers(ra, dec, edgecolor='yellow', marker='o',  s=30, linestyle= '--', alpha=1)


#%%


#Import fits
plw = aplpy.FITSFigure('SPIRE/PLW_masked.fits', figure = fig, subplot=(2,4,8))

#Greyscale
# gc.show_grayscale()

#Customise colorscale here
plw.show_colorscale(cmap='twilight') 

# Customise ticks
plw.tick_labels.set_font(size='small')


#To check coordinates in plot
hmsdms = from_deg_to_hmsdms (ra, dec)

# x, y = gc.world2pixel(ra, dec)
plw.recenter(ra, dec, radius = 0.01)

#Contours
plw.show_contour('SPIRE/PLW_masked.fits', colors='white', filled= False)

# Customise ticks
plw.tick_labels.set_font(size='small')
plw.tick_labels.hide_y()
plw.axis_labels.hide_y()

#Add grid 
plw.add_grid()
plw.grid.set_color('white')
plw.grid.set_alpha(0.5)

#Add title 
plw.set_title('PLW (500 $\mu$m)')

# Add markers
plw.show_markers(ra, dec, edgecolor='yellow', marker='o', s=30, linestyle= '--', alpha=1)

