# %%
import numpy as np

import matplotlib.pyplot as plt
#plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams["font.size"] = 14

import aplpy

from astropy.io import fits

import sourceprocess as sp

# %%

#ra = 264.90184
#dec = 69.06859

ra = 264.89500
dec = 69.06950

# %%
fig = plt.figure(figsize=(12,12))

fig1 = aplpy.FITSFigure('MIPS/MIPS24.fits', figure=fig, subplot=[0,0.75,0.25,0.25])
fig1.recenter(x=ra, y=dec, radius=0.007)
fig1.show_colorscale(cmap='twilight')
fig1.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig1.tick_labels.hide()
fig1.axis_labels.hide_x()
fig1.axis_labels.hide_y()

fig2 = aplpy.FITSFigure('SPIRE/PSW_masked.fits', figure=fig, subplot=[0,0.5,0.25,0.25])
fig2.recenter(x=ra, y=dec, radius=0.007)
fig2.show_colorscale(cmap='twilight')
fig2.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig2.ticks.hide()
fig2.tick_labels.hide()
fig2.axis_labels.hide_x()
fig2.axis_labels.hide_y()

fig3 = aplpy.FITSFigure('SPIRE/PMW_masked.fits', figure=fig, subplot=[0,0.25,0.25,0.25])
fig3.recenter(x=ra, y=dec, radius=0.007)
fig3.show_colorscale(cmap='twilight')
fig3.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig3.tick_labels.hide()
fig3.axis_labels.hide_x()
fig3.axis_labels.hide_y()

fig4 = aplpy.FITSFigure('SPIRE/PLW_masked.fits', figure=fig, subplot=[0,0,0.25,0.25])
fig4.recenter(x=ra, y=dec, radius=0.007)
fig4.show_colorscale(cmap='twilight')
fig4.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
#fig4.ticks.hide()
#fig4.tick_labels.hide()
#fig4.axis_labels.hide_x()
#fig4.axis_labels.hide_y()

#fig5 = aplpy.FITSFigure('IRAC/IRAC1.fits', figure=fig, subplot=[0.25,0.75,0.25,0.25])
#fig5.recenter(x=ra, y=dec, radius=0.007)
#fig5.show_colorscale(cmap='twilight')
#fig5.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
#fig5.ticks.hide()
#fig5.tick_labels.hide()
#fig5.axis_labels.hide_x()
#fig5.axis_labels.hide_y()

fig5 = aplpy.FITSFigure('HST/25704051_HST.fits', figure=fig, subplot=[0.25,0.75,0.25,0.25])
fig5.recenter(x=ra, y=dec, radius=0.007)
fig5.show_colorscale(cmap='twilight')
fig5.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig5.ticks.hide()
fig5.tick_labels.hide()
fig5.axis_labels.hide_x()
fig5.axis_labels.hide_y()

fig6 = aplpy.FITSFigure('IRAC/IRAC1.fits', figure=fig, subplot=[0.25,0.5,0.25,0.25])
fig6.recenter(x=ra, y=dec, radius=0.007)
fig6.show_colorscale(cmap='twilight')
fig6.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig6.ticks.hide()
fig6.tick_labels.hide()
fig6.axis_labels.hide_x()
fig6.axis_labels.hide_y()

fig7 = aplpy.FITSFigure('IRAC/IRAC2.fits', figure=fig, subplot=[0.25,0.5,0.25,0.25])
fig7.recenter(x=ra, y=dec, radius=0.007)
fig7.show_colorscale(cmap='twilight')
fig7.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig7.ticks.hide()
fig7.tick_labels.hide()
fig7.axis_labels.hide_x()
fig7.axis_labels.hide_y()

#fig7 = aplpy.FITSFigure('IRAC/IRAC3.fits', figure=fig, subplot=[0.25,0.25,0.25,0.25])
#fig7.recenter(x=ra, y=dec, radius=0.007)
#fig7.show_colorscale(cmap='twilight')
#fig7.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
#fig7.ticks.hide()
#fig7.tick_labels.hide()
#fig7.axis_labels.hide_x()
#fig7.axis_labels.hide_y()

fig8 = aplpy.FITSFigure('IRAC/IRAC4.fits', figure=fig, subplot=[0.25,0,0.25,0.25])
fig8.recenter(x=ra, y=dec, radius=0.007)
fig8.show_colorscale(cmap='twilight')
fig8.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig8.ticks.hide()
fig8.tick_labels.hide()
fig8.axis_labels.hide_x()
fig8.axis_labels.hide_y()

fig9 = aplpy.FITSFigure('FC_Files-part1/fc_264.902842+69.067949_wise_1.fits', figure=fig, subplot=[0.5,0.75,0.25,0.25])
fig9.recenter(x=ra, y=dec, radius=0.007)
fig9.show_colorscale(cmap='twilight')
fig9.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig9.ticks.hide()
fig9.tick_labels.hide()
fig9.axis_labels.hide_x()
fig9.axis_labels.hide_y()

fig10 = aplpy.FITSFigure('FC_Files-part1/fc_264.902842+69.067949_wise_2.fits', figure=fig, subplot=[0.5,0.5,0.25,0.25])
fig10.recenter(x=ra, y=dec, radius=0.007)
fig10.show_colorscale(cmap='twilight')
fig10.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig10.ticks.hide()
fig10.tick_labels.hide()
fig10.axis_labels.hide_x()
fig10.axis_labels.hide_y()

fig11 = aplpy.FITSFigure('FC_Files-part1/fc_264.902842+69.067949_wise_3.fits', figure=fig, subplot=[0.5,0.25,0.25,0.25])
fig11.recenter(x=ra, y=dec, radius=0.007)
fig11.show_colorscale(cmap='twilight')
fig11.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig11.ticks.hide()
fig11.tick_labels.hide()
fig11.axis_labels.hide_x()
fig11.axis_labels.hide_y()

fig12 = aplpy.FITSFigure('FC_Files-part1/fc_264.902842+69.067949_wise_4.fits', figure=fig, subplot=[0.5,0,0.25,0.25])
fig12.recenter(x=ra, y=dec, radius=0.007)
fig12.show_colorscale(cmap='twilight')
fig12.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig12.ticks.hide()
fig12.tick_labels.hide()
fig12.axis_labels.hide_x()
fig12.axis_labels.hide_y()

#fig13 = aplpy.FITSFigure('HST/25704051_HST.fits', figure=fig, subplot=[0.25,0,0.25,0.25])
#fig13.recenter(x=ra, y=dec, radius=0.006)
#fig13.show_colorscale(cmap='twilight')
#fig13.show_contour('MIPS/MIPS24.fits', colors='white', filled=False, levels=[2.9, 3.], overlap=True)
#fig13.ticks.hide()
#fig13.tick_labels.hide()
#fig13.axis_labels.hide_x()
#fig13.axis_labels.hide_y()

fig.show()

# %%

fig = plt.figure(figsize=(12,12))

fig1 = aplpy.FITSFigure('SPIRE/PSW_masked.fits', figure=fig, subplot=[0,0.5,0.25,0.25])
fig1.recenter(x=ra, y=dec, radius=0.007)
fig1.show_grayscale()
fig1.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
#fig1.ticks.hide_x()
fig1.tick_labels.hide_x()
fig1.axis_labels.hide_x()
fig1.axis_labels.set_ytext('Dec (ICRS)')


fig2 = aplpy.FITSFigure('SPIRE/PMW_masked.fits', figure=fig, subplot=[0,0.25,0.25,0.25])
fig2.recenter(x=ra, y=dec, radius=0.007)
fig2.show_grayscale()
fig2.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
#fig1.ticks.hide()
fig2.tick_labels.hide_x()
fig2.axis_labels.hide_x()
fig2.axis_labels.set_ytext('Dec (ICRS)')

fig3 = aplpy.FITSFigure('SPIRE/PLW_masked.fits', figure=fig, subplot=[0,0,0.25,0.25])
fig3.recenter(x=ra, y=dec, radius=0.007)
fig3.show_grayscale()
fig3.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig3.tick_labels.set_xposition('bottom')
fig3.axis_labels.set_xtext('RA (ICRS)')
fig3.axis_labels.set_ytext('Dec (ICRS)')

fig4 = aplpy.FITSFigure('MIPS/MIPS24.fits', figure=fig, subplot=[0.25,0.5,0.25,0.25])
fig4.recenter(x=ra, y=dec, radius=0.007)
fig4.show_grayscale()
fig4.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
#fig4.ticks.hide()
fig4.tick_labels.hide()
fig4.axis_labels.hide()

fig5 = aplpy.FITSFigure('IRAC/IRAC1.fits', figure=fig, subplot=[0.25,0.25,0.25,0.25])
fig5.recenter(x=ra, y=dec, radius=0.007)
fig5.show_grayscale()
fig5.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
#fig5.ticks.hide()
fig5.tick_labels.hide()
fig5.axis_labels.hide()

fig6 = aplpy.FITSFigure('IRAC/IRAC4.fits', figure=fig, subplot=[0.25,0,0.25,0.25])
fig6.recenter(x=ra, y=dec, radius=0.007)
fig6.show_grayscale()
fig6.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
#fig6.ticks.hide_y()
fig6.tick_labels.hide_y()
fig6.axis_labels.hide_y()
fig6.tick_labels.set_xposition('bottom')
fig6.axis_labels.set_xtext('RA (ICRS)')

fig7 = aplpy.FITSFigure('HST/25704051_HST.fits', figure=fig, subplot=[0.5,0.5,0.25,0.25])
fig7.recenter(x=ra, y=dec, radius=0.007)
fig7.show_grayscale()
fig7.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
#fig7.ticks.hide()
fig7.tick_labels.hide()
fig7.axis_labels.hide()
plt.annotate(text='HST',xy=(0.8,0.2),xycoords='figure fraction', zorder=0)

fig8 = aplpy.FITSFigure('FC_Files-part1/fc_264.902842+69.067949_wise_1.fits', figure=fig, subplot=[0.5,0.25,0.25,0.25])
fig8.recenter(x=ra, y=dec, radius=0.007)
fig8.show_grayscale()
fig8.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
#fig8.ticks.hide()
fig8.tick_labels.hide()
fig8.axis_labels.hide()

fig9 = aplpy.FITSFigure('FC_Files-part1/fc_264.902842+69.067949_wise_2.fits', figure=fig, subplot=[0.5,0,0.25,0.25])
fig9.recenter(x=ra, y=dec, radius=0.007)
fig9.show_grayscale()
fig9.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
#fig9.ticks.hide_y()
fig9.tick_labels.hide_y()
fig9.axis_labels.hide_y()
fig9.tick_labels.set_xposition('bottom')
fig9.axis_labels.set_xtext('RA (ICRS)')

fig.show()

# %%

ra = 264.90270
dec = 69.06810

fig = plt.figure(figsize=(6,6))

fig = aplpy.FITSFigure('HST/25704051_HST.fits', figure=fig)
fig.recenter(x=ra, y=dec, radius=0.0015)
fig.show_grayscale()
fig.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig.axis_labels.set_ytext('Dec (ICRS)')
fig.axis_labels.set_xtext('RA (ICRS)')

# %%
ra = 264.8910
dec = 69.07227

fig = plt.figure(figsize=(6,6))

fig = aplpy.FITSFigure('HST/25704051_HST.fits', figure=fig)
fig.recenter(x=ra, y=dec, radius=0.002)
fig.show_grayscale()
fig.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig.axis_labels.set_ytext('Dec (ICRS)')
fig.axis_labels.set_xtext('RA (ICRS)')

# %%
ra = 264.8935
dec = 69.06860

fig = plt.figure(figsize=(6,6))

fig = aplpy.FITSFigure('HST/25704051_HST.fits', figure=fig)
fig.recenter(x=ra, y=dec, radius=0.0015)
fig.show_grayscale()
fig.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig.axis_labels.set_ytext('Dec (ICRS)')
fig.axis_labels.set_xtext('RA (ICRS)')

# %%

fig1 = aplpy.FITSFigure('HST/25704051_HST.fits')
fig1.show_grayscale()
fig2 = aplpy.FITSFigure('MIPS/MIPS24.fits')
fig2.show_grayscale()

# %%

fig = plt.figure(figsize=(12,12))

fig1 = aplpy.FITSFigure('SPIRE/PSW_masked.fits', figure=fig, subplot=[0,0.5,0.25,0.25])
fig1.recenter(x=ra, y=dec, radius=0.007)
fig1.show_grayscale()
fig1.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, overlap=True)
fig1.ticks.hide_x()
fig1.tick_labels.hide_x()
fig1.axis_labels.hide_x()
fig1.axis_labels.set_ytext('Dec (ICRS)')


fig2 = aplpy.FITSFigure('SPIRE/PMW_masked.fits', figure=fig, subplot=[0,0.25,0.25,0.25])
fig2.recenter(x=ra, y=dec, radius=0.007)
fig2.show_grayscale()
fig2.show_contour('MIPS/MIPS24.fits', colors='red', linewidth=1, filled=False, overlap=True)
fig1.ticks.hide()
fig2.tick_labels.hide_x()
fig2.axis_labels.hide_x()
fig2.axis_labels.set_ytext('Dec (ICRS)')

fig3 = aplpy.FITSFigure('SPIRE/PLW_masked.fits', figure=fig, subplot=[0,0,0.25,0.25])
fig3.recenter(x=ra, y=dec, radius=0.007)
fig3.show_grayscale()
fig3.show_contour('MIPS/MIPS24.fits', colors='red', linewidth=1, filled=False, overlap=True)
fig3.tick_labels.set_xposition('bottom')
fig3.axis_labels.set_xtext('RA (ICRS)')
fig3.axis_labels.set_ytext('Dec (ICRS)')

fig4 = aplpy.FITSFigure('MIPS/MIPS24.fits', figure=fig, subplot=[0.25,0.5,0.25,0.25])
fig4.recenter(x=ra, y=dec, radius=0.007)
fig4.show_grayscale()
fig4.show_contour('MIPS/MIPS24.fits', colors='red', linewidth=1, filled=False, overlap=True)
fig4.ticks.hide()
fig4.tick_labels.hide()
fig4.axis_labels.hide()

fig5 = aplpy.FITSFigure('IRAC/IRAC1.fits', figure=fig, subplot=[0.25,0.25,0.25,0.25])
fig5.recenter(x=ra, y=dec, radius=0.007)
fig5.show_grayscale()
fig5.show_contour('MIPS/MIPS24.fits', colors='red', linewidth=1, filled=False, overlap=True)
fig5.ticks.hide()
fig5.tick_labels.hide()
fig5.axis_labels.hide()

fig6 = aplpy.FITSFigure('IRAC/IRAC4.fits', figure=fig, subplot=[0.25,0,0.25,0.25])
fig6.recenter(x=ra, y=dec, radius=0.007)
fig6.show_grayscale()
fig6.show_contour('MIPS/MIPS24.fits', colors='red', linewidth=1, filled=False, overlap=True)
fig6.ticks.hide_y()
fig6.tick_labels.hide_y()
fig6.axis_labels.hide_y()
fig6.tick_labels.set_xposition('bottom')
fig6.axis_labels.set_xtext('RA (ICRS)')

fig.show()
# %%

# %%

ra = 264.91241
dec = 69.14450

fig = plt.figure(figsize=(12,12))

fig1 = aplpy.FITSFigure('SPIRE/PSW_masked.fits', figure=fig, subplot=[0,0.5,0.25,0.25])
fig1.recenter(x=ra, y=dec, radius=0.006)
fig1.show_grayscale()
fig1.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, overlap=True)
#fig1.ticks.hide_x()
fig1.tick_labels.hide_x()
fig1.axis_labels.hide_x()
fig1.axis_labels.set_ytext('Dec (ICRS)')


fig2 = aplpy.FITSFigure('SPIRE/PMW_masked.fits', figure=fig, subplot=[0,0.25,0.25,0.25])
fig2.recenter(x=ra, y=dec, radius=0.006)
fig2.show_grayscale()
fig2.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, overlap=True)
#fig1.ticks.hide()
fig2.tick_labels.hide_x()
fig2.axis_labels.hide_x()
fig2.axis_labels.set_ytext('Dec (ICRS)')

fig3 = aplpy.FITSFigure('SPIRE/PLW_masked.fits', figure=fig, subplot=[0,0,0.25,0.25])
fig3.recenter(x=ra, y=dec, radius=0.006)
fig3.show_grayscale()
fig3.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, overlap=True)
fig3.tick_labels.set_xposition('bottom')
fig3.axis_labels.set_xtext('RA (ICRS)')
fig3.axis_labels.set_ytext('Dec (ICRS)')

fig4 = aplpy.FITSFigure('MIPS/MIPS24.fits', figure=fig, subplot=[0.25,0.5,0.25,0.25])
fig4.recenter(x=ra, y=dec, radius=0.006)
fig4.show_grayscale()
fig4.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, overlap=True)
#fig4.ticks.hide()
fig4.tick_labels.hide()
fig4.axis_labels.hide()

fig5 = aplpy.FITSFigure('IRAC/IRAC1.fits', figure=fig, subplot=[0.25,0.25,0.25,0.25])
fig5.recenter(x=ra, y=dec, radius=0.006)
fig5.show_grayscale()
fig5.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, overlap=True)
#fig5.ticks.hide()
fig5.tick_labels.hide()
fig5.axis_labels.hide()

fig6 = aplpy.FITSFigure('IRAC/IRAC4.fits', figure=fig, subplot=[0.25,0,0.25,0.25])
fig6.recenter(x=ra, y=dec, radius=0.006)
fig6.show_grayscale()
fig6.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, overlap=True)
#fig6.ticks.hide_y()
fig6.tick_labels.hide_y()
fig6.axis_labels.hide_y()
fig6.tick_labels.set_xposition('bottom')
fig6.axis_labels.set_xtext('RA (ICRS)')

fig.show()

# %%

def niceplot(ra, dec):

    fig = plt.figure(figsize=(12,12))

    fig7 = aplpy.FITSFigure('HST/25704051_HST.fits', figure=fig, subplot=[0.5,0.5,0.25,0.25])
    fig7.recenter(x=ra, y=dec, radius=0.004)
    fig7.show_grayscale()
    fig7.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
    fig7.ticks.hide()
    fig7.tick_labels.hide()
    fig7.axis_labels.hide()

    fig1 = aplpy.FITSFigure('SPIRE/PSW_masked.fits', figure=fig, subplot=[0,0.5,0.25,0.25])
    fig1.recenter(x=ra, y=dec, radius=0.004)
    fig1.show_grayscale()
    fig1.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
    #fig1.ticks.hide_x()
    fig1.tick_labels.hide_x()
    fig1.axis_labels.hide_x()
    fig1.axis_labels.set_ytext('Dec (ICRS)')

    fig2 = aplpy.FITSFigure('SPIRE/PMW_masked.fits', figure=fig, subplot=[0,0.25,0.25,0.25])
    fig2.recenter(x=ra, y=dec, radius=0.004)
    fig2.show_grayscale()
    fig2.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
    #fig1.ticks.hide()
    fig2.tick_labels.hide_x()
    fig2.axis_labels.hide_x()
    fig2.axis_labels.set_ytext('Dec (ICRS)')

    fig3 = aplpy.FITSFigure('SPIRE/PLW_masked.fits', figure=fig, subplot=[0,0,0.25,0.25])
    fig3.recenter(x=ra, y=dec, radius=0.004)
    fig3.show_grayscale()
    fig3.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
    fig3.tick_labels.set_xposition('bottom')
    fig3.axis_labels.set_xtext('RA (ICRS)')
    fig3.axis_labels.set_ytext('Dec (ICRS)')

    fig4 = aplpy.FITSFigure('MIPS/MIPS24.fits', figure=fig, subplot=[0.25,0.5,0.25,0.25])
    fig4.recenter(x=ra, y=dec, radius=0.004)
    fig4.show_grayscale()
    fig4.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
    #fig4.ticks.hide()
    fig4.tick_labels.hide()
    fig4.axis_labels.hide()

    fig5 = aplpy.FITSFigure('IRAC/IRAC1.fits', figure=fig, subplot=[0.25,0.25,0.25,0.25])
    fig5.recenter(x=ra, y=dec, radius=0.004)
    fig5.show_grayscale()
    fig5.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
    #fig5.ticks.hide()
    fig5.tick_labels.hide()
    fig5.axis_labels.hide()

    fig6 = aplpy.FITSFigure('IRAC/IRAC2.fits', figure=fig, subplot=[0.25,0,0.25,0.25])
    fig6.recenter(x=ra, y=dec, radius=0.004)
    fig6.show_grayscale()
    fig6.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
    #fig6.ticks.hide_y()
    fig6.tick_labels.hide_y()
    fig6.axis_labels.hide_y()
    fig6.tick_labels.set_xposition('bottom')
    fig6.axis_labels.set_xtext('RA (ICRS)')

    fig8 = aplpy.FITSFigure('FC_Files-part1/fc_264.902842+69.067949_wise_1.fits', figure=fig, subplot=[0.5,0.25,0.25,0.25])
    fig8.recenter(x=ra, y=dec, radius=0.004)
    fig8.show_grayscale()
    fig8.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
    #fig8.ticks.hide()
    fig8.tick_labels.hide()
    fig8.axis_labels.hide()

    fig9 = aplpy.FITSFigure('FC_Files-part1/fc_264.902842+69.067949_wise_2.fits', figure=fig, subplot=[0.5,0,0.25,0.25])
    fig9.recenter(x=ra, y=dec, radius=0.004)
    fig9.show_grayscale()
    fig9.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
    #fig9.ticks.hide_y()
    fig9.tick_labels.hide_y()
    fig9.axis_labels.hide_y()
    fig9.tick_labels.set_xposition('bottom')
    fig9.axis_labels.set_xtext('RA (ICRS)')

    return fig

    #fig.show()

# %%

import sourceprocess as sp

cat_IRAC = sp.Catalogue('IRAC')
cat_IRAC.get_data('catalogues/IRACdark-matched.csv')

s35668 = sp.Source(id=35668)
s35668.get_position(cat_IRAC)

fig = niceplot(ra=264.81, dec=69.1112)
fig.show()

# %%

ra = 264.813
dec = 69.1125

fig = plt.figure(figsize=(6,6))

fig = aplpy.FITSFigure('HST/25704051_HST.fits', figure=fig)
fig.recenter(x=ra, y=dec, radius=0.0015)
fig.show_grayscale()
fig.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig.axis_labels.set_ytext('Dec (ICRS)')
fig.axis_labels.set_xtext('RA (ICRS)')

# %%
ra = 264.803
dec = 69.1105

fig = plt.figure(figsize=(6,6))

fig = aplpy.FITSFigure('HST/25704051_HST.fits', figure=fig)
fig.recenter(x=ra, y=dec, radius=0.002)
fig.show_grayscale()
fig.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig.axis_labels.set_ytext('Dec (ICRS)')
fig.axis_labels.set_xtext('RA (ICRS)')

# %%

cat_IRAC = sp.Catalogue('IRAC')
cat_IRAC.get_data('catalogues/IRACdark-matched.csv')

s1 = sp.Source(ra=264.813, dec=69.1125) 
s1.get_id(cat_IRAC)

s2 = sp.Source(ra=264.803, dec=69.1105)
s2.get_id(cat_IRAC)

# %%
cat_overlap = sp.Catalogue('HST_overlap')
cat_overlap.get_data('HST_overlap.txt')

sorted_cat_overlap = cat_overlap.data.sort_values('MIPS24 Flux (mJy)')

# %%

for i in range(50,60):

    id = sorted_cat_overlap['ID'].iloc[i]
    print('plotting source', id)
    ra = sorted_cat_overlap['RA'].iloc[i]
    dec = sorted_cat_overlap['Dec'].iloc[i]

    try:
        fig = niceplot(ra, dec)
        fig.show()
    except:
        print('An error has occured for source', id, '(likely not in the picture)')
   
# %%

fig = aplpy.FITSFigure('HST/25704051_HST.fits')
fig.show_grayscale()

# %%
 
ra = 264.858 
dec = 69.0710 

fig = niceplot(ra, dec)
fig.show()

# %%

fig = plt.figure(figsize=(6,6))

fig = aplpy.FITSFigure('HST/25704051_HST.fits', figure=fig)
fig.recenter(x=ra, y=dec, radius=0.0015)
fig.show_grayscale()
fig.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig.axis_labels.set_ytext('Dec (ICRS)')
fig.axis_labels.set_xtext('RA (ICRS)')

# %%

def niceplot_2(ra, dec):

    fig = plt.figure(figsize=(12,12))

    fig7 = aplpy.FITSFigure('HST/25704051_HST.fits', figure=fig, subplot=[0.5,0.5,0.25,0.25])
    fig7.recenter(x=ra, y=dec, radius=0.004)
    fig7.show_grayscale()
    fig7.show_contour('IRAC/IRAC1.fits', colors='red', filled=False, levels=5, overlap=True)
    fig7.ticks.hide()
    fig7.tick_labels.hide()
    fig7.axis_labels.hide()

    fig1 = aplpy.FITSFigure('SPIRE/PSW_masked.fits', figure=fig, subplot=[0,0.5,0.25,0.25])
    fig1.recenter(x=ra, y=dec, radius=0.004)
    fig1.show_grayscale()
    fig1.show_contour('IRAC/IRAC1.fits', colors='red', filled=False, levels=5, overlap=True)
    #fig1.ticks.hide_x()
    fig1.tick_labels.hide_x()
    fig1.axis_labels.hide_x()
    fig1.axis_labels.set_ytext('Dec (ICRS)')

    fig2 = aplpy.FITSFigure('SPIRE/PMW_masked.fits', figure=fig, subplot=[0,0.25,0.25,0.25])
    fig2.recenter(x=ra, y=dec, radius=0.004)
    fig2.show_grayscale()
    fig2.show_contour('IRAC/IRAC1.fits', colors='red', filled=False, levels=5, overlap=True)
    #fig1.ticks.hide()
    fig2.tick_labels.hide_x()
    fig2.axis_labels.hide_x()
    fig2.axis_labels.set_ytext('Dec (ICRS)')

    fig3 = aplpy.FITSFigure('SPIRE/PLW_masked.fits', figure=fig, subplot=[0,0,0.25,0.25])
    fig3.recenter(x=ra, y=dec, radius=0.004)
    fig3.show_grayscale()
    fig3.show_contour('IRAC/IRAC1.fits', colors='red', filled=False, levels=5, overlap=True)
    fig3.tick_labels.set_xposition('bottom')
    fig3.axis_labels.set_xtext('RA (ICRS)')
    fig3.axis_labels.set_ytext('Dec (ICRS)')

    fig4 = aplpy.FITSFigure('MIPS/MIPS24.fits', figure=fig, subplot=[0.25,0.5,0.25,0.25])
    fig4.recenter(x=ra, y=dec, radius=0.004)
    fig4.show_grayscale()
    fig4.show_contour('IRAC/IRAC1.fits', colors='red', filled=False, levels=5, overlap=True)
    #fig4.ticks.hide()
    fig4.tick_labels.hide()
    fig4.axis_labels.hide()

    fig5 = aplpy.FITSFigure('IRAC/IRAC1.fits', figure=fig, subplot=[0.25,0.25,0.25,0.25])
    fig5.recenter(x=ra, y=dec, radius=0.004)
    fig5.show_grayscale()
    fig5.show_contour('IRAC/IRAC1.fits', colors='red', filled=False, levels=5, overlap=True)
    #fig5.ticks.hide()
    fig5.tick_labels.hide()
    fig5.axis_labels.hide()

    fig6 = aplpy.FITSFigure('IRAC/IRAC2.fits', figure=fig, subplot=[0.25,0,0.25,0.25])
    fig6.recenter(x=ra, y=dec, radius=0.004)
    fig6.show_grayscale()
    fig6.show_contour('IRAC/IRAC1.fits', colors='red', filled=False, levels=5, overlap=True)
    #fig6.ticks.hide_y()
    fig6.tick_labels.hide_y()
    fig6.axis_labels.hide_y()
    fig6.tick_labels.set_xposition('bottom')
    fig6.axis_labels.set_xtext('RA (ICRS)')

    fig8 = aplpy.FITSFigure('FC_Files-part1/fc_264.902842+69.067949_wise_1.fits', figure=fig, subplot=[0.5,0.25,0.25,0.25])
    fig8.recenter(x=ra, y=dec, radius=0.004)
    fig8.show_grayscale()
    fig8.show_contour('IRAC/IRAC1.fits', colors='red', filled=False, levels=5, overlap=True)
    #fig8.ticks.hide()
    fig8.tick_labels.hide()
    fig8.axis_labels.hide()

    fig9 = aplpy.FITSFigure('FC_Files-part1/fc_264.902842+69.067949_wise_2.fits', figure=fig, subplot=[0.5,0,0.25,0.25])
    fig9.recenter(x=ra, y=dec, radius=0.004)
    fig9.show_grayscale()
    fig9.show_contour('IRAC/IRAC1.fits', colors='red', filled=False, levels=5, overlap=True)
    #fig9.ticks.hide_y()
    fig9.tick_labels.hide_y()
    fig9.axis_labels.hide_y()
    fig9.tick_labels.set_xposition('bottom')
    fig9.axis_labels.set_xtext('RA (ICRS)')

    return fig


# %%

ra = 264.8324473
dec = 69.0912856

fig = niceplot_2(ra, dec)
fig.show()

# %%

fig = plt.figure(figsize=(6,6))

fig = aplpy.FITSFigure('HST/25704051_HST.fits', figure=fig)
fig.recenter(x=ra, y=dec, radius=0.0015)
fig.show_grayscale()
fig.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig.axis_labels.set_ytext('Dec (ICRS)')
fig.axis_labels.set_xtext('RA (ICRS)')

# %%

ra = 264.8249330
dec = 69.1031363

# %%
fig = niceplot_2(ra, dec)
fig.show()

# %%

fig = plt.figure(figsize=(6,6))

fig = aplpy.FITSFigure('HST/25704051_HST.fits', figure=fig)
fig.recenter(x=ra, y=dec, radius=0.0015)
fig.show_grayscale()
fig.show_contour('MIPS/MIPS24.fits', colors='red', filled=False, levels=[2.8, 2.9, 3.], overlap=True)
fig.axis_labels.set_ytext('Dec (ICRS)')
fig.axis_labels.set_xtext('RA (ICRS)')

# %%

ra = 264.7990202
dec = 69.0978651

fig = niceplot_2(ra, dec)
fig.show()

# %%
fig = plt.figure(figsize=(6,6))

fig = aplpy.FITSFigure('HST/25704051_HST.fits', figure=fig)
fig.recenter(x=ra, y=dec, radius=0.001)
fig.show_grayscale()
fig.show_contour('IRAC/IRAC1.fits', colors='red', filled=False, levels=5, overlap=True)
fig.axis_labels.set_ytext('Dec (ICRS)')
fig.axis_labels.set_xtext('RA (ICRS)')

# %%

ra = 264.858 
dec = 69.0710 

fig = niceplot_2(ra, dec)
fig.show()
# %%
