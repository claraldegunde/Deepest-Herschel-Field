#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:54:30 2023

@author: claraaldegundemanteca
"""

import aplpy
# documentation: https://buildmedia.readthedocs.org/media/pdf/aplpy/v0.9.11/aplpy.pdf


#Import fits
gc = aplpy.FITSFigure('SPIRE/PSW_masked.fits')

#Greyscale
# gc.show_grayscale()

#Customise colorscale here
gc.show_colorscale(cmap='viridis') 

#Generate rgb image using a png containing the 3 color imag
# gc.show_rgb('.png') 

# Customise font
# gc.tick_labels.set_font(size='small')

#Contours
# gc.show_contour('SPIRE/PSW_masked.fits', colors='r', filled= False)

#Add grid 
gc.add_grid()
gc.grid.set_color('k')

# Add markers
gc.show_markers([265.68493],[69], edgecolor='r', marker='o', s=10, alpha=1)

#Convert pixel coordinates to world coordinates 
gc.pixel2world(200,200) #x, y pixel coordinates

