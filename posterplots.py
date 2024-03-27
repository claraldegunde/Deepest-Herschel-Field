# %%

from PIL import Image

from astropy.io import fits

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import aplpy

# %%

# MAKE TRANSPARENT DEEP FIELD
img = Image.open('deepfield4.png')
rgba = img.convert("RGBA")
datas = rgba.getdata()

newData = []
for item in datas:
    if item[0] == 255 and item[1] == 255 and item[2] == 255:
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)
rgba.putdata(newData)
rgba.save("transparent_deepfield4.png", "PNG")

# %%

# IMPORT AVENIR FONT

import matplotlib.font_manager as font_manager

font_dir = ["C:/Users/bonac/Downloads/Avenir-Font/avenir_ff"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

# %%

# OBSERVATIONS

hdulist_obs = fits.open('observations.fits')
table_obs = hdulist_obs[1].data

df_obs = pd.DataFrame()
df_obs["id"] = table_obs["id"]
df_obs["redshift"] = table_obs["redshift"]
df_obs["PSW"] = table_obs["PSW"]
df_obs["PMW"] = table_obs["PMW"]
df_obs["PLW"] = table_obs["PLW"]
df_obs["IRAC1"] = table_obs["IRAC1"]
df_obs["IRAC2"] = table_obs["IRAC2"]
df_obs["IRAC3"] = table_obs["IRAC3"]
df_obs["IRAC4"] = table_obs["IRAC4"]
df_obs["MIPS1"] = table_obs["MIPS1"]
df_obs["u_prime"] = table_obs["u_prime"]
df_obs["g_prime"] = table_obs["g_prime"]
df_obs["r_prime"] = table_obs["r_prime"]
df_obs["z_prime"] = table_obs["z_prime"]

df_obs_err = pd.DataFrame()
df_obs_err["id"] = table_obs["id"]
df_obs_err["redshift"] = table_obs["redshift"]
df_obs_err["PSW-err"] = table_obs["PSW_err"]
df_obs_err["PMW_err"] = table_obs["PMW_err"]
df_obs_err["PLW_err"] = table_obs["PLW_err"]
df_obs_err["IRAC1_err"] = table_obs["IRAC1_err"]
df_obs_err["IRAC2_err"] = table_obs["IRAC2_err"]
df_obs_err["IRAC3_err"] = table_obs["IRAC3_err"]
df_obs_err["IRAC4_err"] = table_obs["IRAC4_err"]
df_obs_err["MIPS1_err"] = table_obs["MIPS1_err"]
df_obs_err["u_prime_err"] = table_obs["u_prime_err"]
df_obs_err["g_prime_err"] = table_obs["g_prime_err"]
df_obs_err["r_prime_err"] = table_obs["r_prime_err"]
df_obs_err["z_prime_err"] = table_obs["z_prime_err"]

# SED SOURCE 3124
obs_3124 = df_obs.iloc[0].values[2:]
obs_err_3124 = df_obs_err.iloc[0].values[2:]
lamb_obs_3124 = np.array([250, 350, 500, 3.6, 4.6, 5.8, 8, 24, 265e-3, 472e-3, 641.5e-3, 926e-3]) #microns

hdulist_sed = fits.open("3124_best_model.fits")
table_sed = hdulist_sed[1].data
lamb = table_sed['wavelength']*1e-3 #microns
sed = table_sed['Fnu']
stellatt = table_sed['attenuation.stellar.young']
stellunatt = table_sed['stellar.young']
dustatt = table_sed['dust.Umin_Umax']
igm = table_sed['igm']
L = table_sed['L_lambda_total']

# observed fluxes
hdulist_obs = fits.open('observations.fits')
table_obs = hdulist_obs[1].data

plt.rcParams["font.family"] = "Arial"
plt.rcParams ["mathtext.default"] = 'regular'
fig, ax = plt.subplots(figsize=(8,4))

ax.scatter(lamb_obs_3124[-4:], obs_3124[-4:], s=50, edgecolors='#97FEED', facecolor='#97FEED', label='LFC', zorder=2)
ax.scatter(lamb_obs_3124[3:-4], obs_3124[3:-4], s=50, edgecolors='#9EDE73', facecolor='#9EDE73', label='IRAC', zorder=1)
ax.scatter(lamb_obs_3124[2], obs_3124[2], s=50, edgecolors='#FFC600', facecolor='#FFC600', label='SPIRE', zorder=3)
ax.scatter(lamb_obs_3124[1], obs_3124[1], s=50, edgecolors='#FFC600', facecolor='#FFC600', zorder=4)
ax.scatter(lamb_obs_3124[0], obs_3124[0], s=50, edgecolors='#FFC600', facecolor='#FFC600', zorder=5)
ax.errorbar(lamb_obs_3124[3:-4], obs_3124[3:-4], obs_err_3124[3:-4], linestyle=' ', elinewidth=1.5, marker='none', color='#9EDE73', capsize=3, capthick=1.5, zorder=1)
ax.errorbar(lamb_obs_3124[-4:], obs_3124[-4:], obs_err_3124[-4:], linestyle=' ', elinewidth=1.5, marker='none', color='#97FEED', capsize=3, capthick=1.5, zorder=2)
ax.errorbar(lamb_obs_3124[2], obs_3124[2], obs_err_3124[2], linestyle=' ', elinewidth=1.5, marker='none', color='#FFC600', capthick=1.5, zorder=3)
ax.errorbar(lamb_obs_3124[1], obs_3124[1], obs_err_3124[1], linestyle=' ', elinewidth=1.5, marker='none', color='#FFC600', capthick=1.5, zorder=4)
ax.errorbar(lamb_obs_3124[0], obs_3124[0], obs_err_3124[0], linestyle=' ', elinewidth=1.5, marker='none', color='#FFC600', capthick=1.5, zorder=5)
ax.plot(lamb, sed, linewidth=1, color='white', zorder=0, label='best fit model')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim((min(lamb_obs_3124)-(min(lamb_obs_3124))*2e-1, max(lamb_obs_3124)+(max(lamb_obs_3124))*5e-1))
ax.set_ylim((min(obs_3124)-(min(obs_3124))*8e-1, max(obs_3124)+(max(obs_3124))*8e-1))
#ax.tick_params(top=False, right=False, left=False, bottom=False)
ax.tick_params(axis='both', which='major', direction='in', colors='white', width=1, labelsize=14, top=True, right=True, left=True, bottom=True)
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.xaxis.label.set_color('white')
ax.set_xlabel(r'Wavelength $\lambda\;[\mu m]$', color='white', fontsize=16, font='Arial')
ax.set_ylabel(r'Flux $S_{\nu}\;[mJy]$', color='white', fontsize=16, font='Arial')
ax.legend(labelcolor='white', fontsize=14, facecolor='none', loc='lower right', framealpha=0.2)
#fig.set_facecolor('#141022ff')
#ax.set_facecolor('#141022ff')
fig.tight_layout()
fig.savefig('SED3124version8.png', transparent=True, dpi=300)
fig.show()

# %%

# SED SOURCE 2367                                                                
obs_2367 = df_obs.iloc[3].values[2:]
obs_err_2367 = df_obs_err.iloc[3].values[2:]
lamb_obs_2367 = np.array([250, 350, 500, 3.6, 4.6, 5.8, 8, 24, 265e-3, 472e-3, 641.5e-3, 926e-3]) #microns

hdulist_sed = fits.open("2367_best_model.fits")
table_sed = hdulist_sed[1].data
lamb = table_sed['wavelength']*1e-3 #microns
sed = table_sed['Fnu']
stellatt = table_sed['attenuation.stellar.young']
stellunatt = table_sed['stellar.young']
dustatt = table_sed['dust.Umin_Umax']
igm = table_sed['igm']
L = table_sed['L_lambda_total']

# observed fluxes
hdulist_obs = fits.open('observations.fits')
table_obs = hdulist_obs[1].data

plt.rcParams["font.family"] = "Arial"
plt.rcParams ["mathtext.default"] = 'regular'
fig, ax = plt.subplots(figsize=(8,4))

ax.scatter(lamb_obs_2367[-4:], obs_2367[-4:], s=50, edgecolors='#97FEED', facecolor='#97FEED', label='LFC', zorder=1)
ax.scatter(lamb_obs_2367[3:-4], obs_2367[3:-4], s=50, edgecolors='#9EDE73', facecolor='#9EDE73', label='IRAC', zorder=2)
ax.scatter(lamb_obs_2367[2], obs_2367[2], s=50, edgecolors='#FFC600', facecolor='#FFC600', label='SPIRE', zorder=3)
ax.scatter(lamb_obs_2367[1], obs_2367[1], s=50, edgecolors='#FFC600', facecolor='#FFC600', zorder=4)
ax.scatter(lamb_obs_2367[0], obs_2367[0], s=50, edgecolors='#FFC600', facecolor='#FFC600', zorder=5)
ax.errorbar(lamb_obs_2367[-4:], obs_2367[-4:], obs_err_2367[-4:], linestyle=' ', elinewidth=1.5, marker='none', color='#97FEED', capsize=3, capthick=1.5, zorder=1)
ax.errorbar(lamb_obs_2367[3:-4], obs_2367[3:-4], obs_err_2367[3:-4], linestyle=' ', elinewidth=1.5, marker='none', color='#9EDE73', capsize=3, capthick=1.5, zorder=2)
ax.errorbar(lamb_obs_2367[2], obs_2367[2], obs_err_2367[2], linestyle=' ', elinewidth=1.5, marker='none', color='#FFC600', capsize=3, capthick=1.5, zorder=3)
ax.errorbar(lamb_obs_2367[1], obs_2367[1], obs_err_2367[1], linestyle=' ', elinewidth=1.5, marker='none', color='#FFC600', capsize=3, capthick=1.5, zorder=4)
ax.errorbar(lamb_obs_2367[0], obs_2367[0], obs_err_2367[0], linestyle=' ', elinewidth=1.5, marker='none', color='#FFC600', capsize=3, capthick=1.5, zorder=5)
ax.plot(lamb, sed, linewidth=1, color='white', label='best fit model', zorder=0)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim((min(lamb_obs_2367)-(min(lamb_obs_2367))*2e-1, max(lamb_obs_2367)+(max(lamb_obs_2367))*5e-1))
ax.set_ylim((min(obs_2367)-(min(obs_2367))*8e-1, max(obs_2367)+(max(obs_2367))*1.5))
ax.tick_params(reset=True, axis='both', which='major', direction='in', colors='white', width=1, labelsize=14, top=True, right=True)
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.xaxis.label.set_color('white')
ax.set_xlabel(r'Wavelength $\lambda\;[\mu m]$', color='white', fontsize=16, font='Arial')
ax.set_ylabel(r'Flux $S_{\nu}\;[mJy]$', color='white', fontsize=16, font='Arial')
ax.legend(labelcolor='white', fontsize=14, framealpha=0.2, facecolor='none', loc='lower right')
#fig.set_facecolor('#141022ff')
#ax.set_facecolor('#141022ff')
fig.tight_layout()
fig.savefig('SED2367version8.png', transparent=True, dpi=300)
fig.show()

# %%

# SOURCE 3124
ra = 264.903
dec = 69.06810

plt.rcParams["font.family"] = "Arial"
plt.rcParams ["mathtext.default"] = 'regular'

fig1 = plt.figure(figsize=(6,6))
im = aplpy.FITSFigure('HST/25704051_HST.fits', figure=fig1)
im.recenter(x=ra, y=dec, radius=0.001)
im.show_colorscale(cmap='inferno', stretch='power')
im.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.16, 0.3])
im.frame.set_color('white')
im.frame.set_linewidth(1.5)
im.ticks.set_tick_direction('in')
im.ticks.set_color('white')
im.ticks.set_linewidth(1.5)
im.tick_labels.hide()
im.axis_labels.hide()
plt.annotate(text=r'$\lambda = 0.6\; \mu m$', xy=(50,360), xycoords='figure points', color='white', fontsize=24)
#fig1.set_facecolor('#141022ff')
fig1.tight_layout()
fig1.savefig('HST3124.png', transparent=True, dpi=300)
fig1.show()

# %%
fig2 = plt.figure(figsize=(6,6))
im = aplpy.FITSFigure('IRAC/IRAC1.fits', figure=fig2)
im.recenter(x=ra, y=dec, radius=0.001)
im.show_colorscale(cmap='inferno')
im.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.16, 0.3])
im.frame.set_color('white')
im.frame.set_linewidth(1.5)
im.ticks.set_tick_direction('in')
im.ticks.set_color('white')
im.ticks.set_linewidth(1.5)
im.tick_labels.hide()
im.axis_labels.hide()
plt.annotate(text=r'$\lambda = 3.6\;\mu m$', xy=(50,360), xycoords='figure points', color='white', fontsize=24)
#fig2.set_facecolor('#141022ff')
fig2.tight_layout()
fig2.savefig('IRAC3124.png', transparent=True, dpi=300)
fig2.show()

# %%
# SOURCE 2367
ra = 264.803
dec = 69.1103

plt.rcParams["font.family"] = "Arial"
plt.rcParams ["mathtext.default"] = 'regular'

fig1 = plt.figure(figsize=(6,6))
im = aplpy.FITSFigure('HST/25704051_HST.fits', figure=fig1)
im.recenter(x=ra, y=dec, radius=0.0015)
im.show_colorscale(cmap='inferno', stretch='power')
im.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.16, 0.3])
im.frame.set_color('white')
im.frame.set_linewidth(1.5)
im.ticks.set_tick_direction('in')
im.ticks.set_color('white')
im.ticks.set_linewidth(1.5)
im.tick_labels.hide()
im.axis_labels.hide()
plt.annotate(text=r'$\lambda = 0.6\; \mu m$', xy=(50,360), xycoords='figure points', color='white', fontsize=24)
#fig1.set_facecolor('#141022ff')
fig1.tight_layout()
fig1.savefig('HST2367.png', transparent=True, dpi=300)
fig1.show()

# %%
fig2 = plt.figure(figsize=(6,6))
im = aplpy.FITSFigure('IRAC/IRAC2.fits', figure=fig2)
im.recenter(x=ra, y=dec, radius=0.0015)
im.show_colorscale(cmap='inferno')
im.show_contour('IRAC/IRAC1.fits', colors='white', filled=False, levels=[0.16, 0.3])
im.frame.set_color('white')
im.frame.set_linewidth(1.5)
im.ticks.set_tick_direction('in')
im.ticks.set_color('white')
im.ticks.set_linewidth(1.5)
im.tick_labels.hide()
im.axis_labels.hide()
plt.annotate(text=r'$\lambda = 3.6\;\mu m$', xy=(50,360), xycoords='figure points', color='white', fontsize=24)
#fig2.set_facecolor('#141022ff')
fig2.tight_layout()
fig2.savefig('IRAC2367.png', transparent=True, dpi=300)
fig2.show()

# %%
