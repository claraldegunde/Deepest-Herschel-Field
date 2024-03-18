# %%

from PIL import Image

from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np

import pandas as pd

import aplpy

# %%

# SOURCE 3124
ra = 264.903
dec = 69.06810

plt.rcParams["font.family"] = "Arial"
plt.rcParams ["mathtext.default"] = 'regular'

fig1 = plt.figure(figsize=(12,12))
#ax.spines[:].set_visible(False)
#ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

im1 = aplpy.FITSFigure('SPIRE/PSW_masked.fits', figure=fig1, subplot=[0.1,0.1,0.1,0.1])
im1.recenter(x=ra, y=dec, radius=0.005)
im1.show_colorscale(cmap='inferno', stretch='power')
im1.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.9, 3.])
im1.frame.set_linewidth(1)
im1.ticks.set_tick_direction('in')
im1.ticks.set_linewidth(1)
im1.ticks.set_color('white')
im1.tick_labels.hide()
im1.axis_labels.hide()

im2 = aplpy.FITSFigure('MIPS/MIPS24.fits', figure=fig1, subplot=[0.2,0.1,0.1,0.1])
im2.recenter(x=ra, y=dec, radius=0.005)
im2.show_colorscale(cmap='inferno')
im2.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.9, 3.])
im2.frame.set_linewidth(1)
im2.ticks.set_tick_direction('in')
im2.ticks.set_linewidth(1)
im2.ticks.set_color('white')
im2.tick_labels.hide()
im2.axis_labels.hide()

im3 = aplpy.FITSFigure('MIPS/MIPS24.fits', figure=fig1, subplot=[0.33,0.1,0.1,0.1])
im3.recenter(x=ra, y=dec, radius=0.002)
im3.show_colorscale(cmap='inferno')
im3.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.9, 3.])
im3.frame.set_linewidth(1)
im3.ticks.set_tick_direction('in')
im3.ticks.set_linewidth(1)
im3.ticks.set_color('white')
im3.tick_labels.hide()
im3.axis_labels.hide()

im4 = aplpy.FITSFigure('IRAC/IRAC1.fits', figure=fig1, subplot=[0.43,0.1,0.1,0.1])
im4.recenter(x=ra, y=dec, radius=0.002)
im4.show_colorscale(cmap='inferno')
im4.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.9, 3.])
im4.frame.set_linewidth(1)
im4.ticks.set_tick_direction('in')
im4.ticks.set_linewidth(1)
im4.ticks.set_color('white')
im4.tick_labels.hide()
im4.axis_labels.hide()

im6 = aplpy.FITSFigure('HST/25704051_HST.fits', figure=fig1, subplot=[0.66,0.1,0.1,0.1])
im6.recenter(x=ra, y=dec, radius=0.001)
im6.show_colorscale(cmap='inferno', stretch='power')
im6.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.9, 3.])
im6.frame.set_linewidth(1)
im6.ticks.set_tick_direction('in')
im6.ticks.set_linewidth(1)
im6.ticks.set_color('white')
im6.tick_labels.hide()
im6.axis_labels.hide()
#plt.annotate(text=r'$\lambda = 0.6\; \mu m$', xy=(50,360), xycoords='figure points', color='white', fontsize=24)

im5 = aplpy.FITSFigure('IRAC/IRAC1.fits', figure=fig1, subplot=[0.56,0.1,0.1,0.1])
im5.recenter(x=ra, y=dec, radius=0.001)
im5.show_colorscale(cmap='inferno')
im5.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.9, 3.])
im5.frame.set_linewidth(1)
im5.ticks.set_tick_direction('in')
im5.ticks.set_linewidth(1)
im5.ticks.set_color('white')
im5.tick_labels.hide()
im5.axis_labels.hide()
#plt.annotate(text=r'$\lambda = 0.6\; \mu m$', xy=(50,360), xycoords='figure points', color='white', fontsize=24)

axes = fig1.get_axes()
axes[4].add_patch(Rectangle((800, 400), 200, 200, facecolor=None, edgecolor='white', zorder=0))
#fig1.tight_layout()
#fig1.savefig('HST3124.png', transparent=True, dpi=300)
fig1.show()
# %%
