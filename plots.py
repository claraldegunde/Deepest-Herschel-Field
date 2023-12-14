"""
@author: Clarisse Bonacina
"""

"""
A plotting module

Classes
-------

Functions
---------

"""

# %%

import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import aplpy

from astropy import units as u
from astropy.coordinates import SkyCoord

from astroquery.mast import Mast, Observations

# %%

def get_HST_image(ra, dec, r, return_path=False):
    '''
    Performs a cone search through the mast database and downloads corresponding fits file.

    Parameters
    ----------
    ra: float
        ra position of the source in degrees
    dec: float
        dec position of the source in degrees
    r: float
        radius of the cone in degrees
    return_path: bool, default is False
        if True, the function returns the path of the image file
    
    Returns
    -------
    local_path: str
        path of the image file

    '''
    
    try:
        # cone search through hubble 
        service = 'Mast.Caom.Cone'
        params = {'ra':ra, 'dec':dec, 'radius':r}
        obs = Mast.service_request(service, params)

        # download image
        line = obs[obs['obs_collection'] == 'HLA'][0]
        obsid = line['obsid']
        pdcts = Observations.get_product_list(obsid)
        pdcts_fil = Observations.filter_products(pdcts, productType=["SCIENCE"], extension="fits") 
        pdct = pdcts_fil[0]["dataURI"]
        local_path = 'HST/' + obsid + '_HST.fits'
        res = Observations.download_file(pdct, local_path=local_path)
        
        if return_path:
            return local_path

    except:
        print('No image file found for this region of the sky.')


def grid_plot(ra, dec, r, paths, contours=None, levels=5, savefig=False, outputfile='figure.png'):
    '''
    This function plots fits images centred on the same source (beta version, to be improved).

    Parameters
    ----------
    ra: float
        ra position of the source in degrees
    dec: float
        dec position of the source in degrees
    r: float
        radius of the cone in degrees
    paths: list of str
        list containing the paths to the fits files
    contours: str, default is None
        path to the fits file used to evaluate the contours
    levels: int, default is 5
        number of contours to plot (see aplpy documentation)
    savefig: bool, default is False
        if True, the figure will be saved to a png
    outputfile: str, default is 'figure.png'
        output directory

    '''


    cols = 4
    rows = int(np.ceil(len(paths)/4))
    
    fig = plt.figure(figsize=(6*rows,6*cols))
    
    for i, path in enumerate(paths):

        im = aplpy.FITSFigure(path, figure=fig, subplot=(rows, cols, i+1))

        im.show_colorscale(cmap='twilight') 
        im.recenter(x=ra, y=dec, radius=r)
    
        im.add_grid()
        im.grid.set_color('white')
        im.grid.set_alpha(0.5)
        im.tick_labels.set_font(size='small')#
        
        if i != 0 and i != 4 and i != 8:
            im.axis_labels.hide_y()
        
        if i < 8:
            im.axis_labels.hide_x()

        title = (path.split('/'))[1][:-5]
        im.set_title(title)

        if type(contours) == str:
            im.show_contour(contours, colors='white', filled=False, levels=levels, overlap=True)

    if savefig:
        fig.savefig(outputfile)

    fig.show()

# %%
# EXAMPLE CELL

# source 3124
get_HST_image(ra=264.90184, dec=69.06859, r=0.005)

paths = ['MIPS/MIPS24.fits',
         'HST/25704051_HST.fits',
         'SPIRE/PSW_masked.fits',
         'SPIRE/PMW_masked.fits',
         'SPIRE/PLW_masked.fits',
         'IRAC/IRAC1.fits',
         'IRAC/IRAC2.fits',
         'IRAC/IRAC3.fits',
         'IRAC/IRAC4.fits',
         'FC_Files-part1/fc_264.902842+69.067949_wise_1.fits',
         'FC_Files-part1/fc_264.902842+69.067949_wise_2.fits',
         'FC_Files-part1/fc_264.902842+69.067949_wise_3.fits',
         'FC_Files-part1/fc_264.902842+69.067949_wise_4.fits']
grid_plot(ra=264.90184, dec=69.06859, r=0.006, paths=paths, contours='HST/25704051_HST.fits', levels=2, savefig=True, outputfile='3124HSTcontours')
grid_plot(ra=264.90184, dec=69.06859, r=0.006, paths=paths, contours='MIPS/MIPS24.fits', levels=5, savefig=True, outputfile='3124MIPS24contours')
