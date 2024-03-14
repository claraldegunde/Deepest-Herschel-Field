u"""
@author: Clarisse Bonacina
"""

"""
A module to process source data

Classes
-------
Catalogue
    Catalogue Class
Source
    Source class

Functions
---------
from_deg_to_hmsdms
    Converts ra dec position from deg to hmsdms.
from_hmsdms_to_deg
    Converts ra dec position from hmsdms to deg.
ang_sep
    Find the angular separation between two sources (from https://www.skythisweek.info/angsep.pdf)

"""

# %%
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, FK5


def from_m_to_F (F0, m):
    '''
    Converts magnitude to flux F (in the units of F0).

    Parameters
    ----------
    F0 : float
        zero magnitude flux
    m : float
        magnitude of the source
    
    Returns
    -------
    F : float
        flux in the units of F0

    '''

    F = F0*10**(-m/2.5)

    return F


def from_deg_to_hmsdms (ra, dec):
    '''
    Converts ra dec position from deg to hmsdms.

    Parameters
    ----------
    ra : float
        ra position in degrees
    dec : float
        dec position in degrees

    Returns
    -------
    c_str : list of str
        list containing the ra and dec positions in hourangle


    '''
    c = SkyCoord(ra, dec, frame=FK5, unit = 'deg')
    c_str = (c.to_string('hmsdms')).split()

    return c_str


def from_hmsdms_to_deg (ra, dec):
    '''
    Converts ra dec position from hmsdms to deg.

    Parameters
    ----------
    ra : str
        ra position in hour angle
    dec : str
        dec position in hour angle

    Returns
    -------
    c_str : list of str
        list containing the ra and dec positions in degrees

    '''
    c = SkyCoord(ra, dec, frame=FK5, unit = 'hourangle')
    c_str = (c.to_string('decimal')).split()

    return c_str


def ang_sep(ra1, ra2, dec1, dec2):
    '''
    Find the angular separation between two sources (from https://www.skythisweek.info/angsep.pdf)

    Parameters
    ----------
    ra1 : float
        ra position of source 1 in degrees
    ra2 : float
        ra position of source 2 in degrees
    dec1 : float
        dec position of source 1 in degrees
    dec2 : float
        dec position of source 2 in degrees

    Returns
    -------
    sep : float
        angular separation between source 1 and 2 in degrees

    '''
    pref = 180/np.pi
    num = np.sqrt(np.cos(dec2)*(np.sin(ra2-ra1))**2 + (np.cos(dec1)*np.sin(dec2)-np.sin(dec1)*np.cos(dec2)*np.cos(ra2-ra1))**2)
    denom = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra2-ra1)
    sep = np.abs(pref * np.arctan(num/denom))

    return sep



class Catalogue:
    '''
    Catalogue class

    Attributes
    ----------
    name : str
        name to label your catalogue
    data : pandas DataFrame object, default is None
        data from the catalogue
    columns : list of str, default is None
        names of the catalogue columns

    Methods
    -------
    get_data():
        uploads the catalogue data into a dataframe
    '''

    def __init__(self, name):

        self.name = name
    
    def get_data(self, path):

        df = pd.read_csv(path, low_memory=True)

        self.data = df
        self.columns = df.columns

        return df


class Source:
    '''
    Source class

    Attributes
    ----------
    id : int, default is None
        source ID
    ra : float, default is None
        ra position in degrees
    dec : float
        dec position in degrees
    fluxes : pandas DataFrame object
        dataframe containing the id, position, redshift and fluxes of the source

    Methods
    -------
    get_position :

    get_id :

    get_fluxes :

    '''

    def __init__(self, id=None, ra=None, dec=None):

        self.id = id

        if type(ra) == float and type(dec) == float:
            self.ra = ra
            self.dec = dec
        
        elif type(ra) == np.float64 and type(dec) == np.float64:
            self.ra = float(ra)
            self.dec = float(dec)

        elif type(ra) == str and type(dec) == str:
            c_deg = from_hmsdms_to_deg(ra, dec)
            self.ra = float(c_deg[0])
            self.dec = float(c_deg[1])
    def __str__(self):
        
        return f'Source(ID: {self.id}, ra: {self.ra}, dec: {self.dec})'



    def get_position(self, cat):

        data = cat.data

        self.ra = (data[cat.columns[2]][data[cat.columns[0]] == self.id]).iloc[0]
        self.dec = (data[cat.columns[3]][data[cat.columns[0]] == self.id]).iloc[0]

    def get_id(self, cat):
        print(self.ra, self.dec)
        data = cat.data

        ra_col = data[cat.columns[2]]
        dec_col = data[cat.columns[3]]
        sep_list = []
        for i in range (0, len(ra_col)):
            sep_list.append(ang_sep(self.ra, ra_col[i], self.dec, dec_col[i]))
        sep_arr = np.array(sep_list)
        index = np.where(np.nanmin(sep_arr) == sep_arr)[0][0]
        self.id = data[cat.columns[0]].iloc[index]
        
    def get_fluxes(self, cat):
        '''
        Retrieves the fluxes from the catalogue and uploads them to a dataframe.

        Parameters
        ----------
        cat: instance of Catalogue class

        '''

        # check input
        if self.id == None and self.ra == None and self.dec == None:
            raise ValueError("id or source position must be given")
            

        # some fluxes from another catalogue are present
        if hasattr(self, 'fluxes'):

            data = cat.data

            # find source from id
            if hasattr(self, 'id'):

                newfluxes = (data[data[cat.columns[0]] == self.id]).loc[:, cat.columns[4]:]
                
                # break here if id no matching id found in catalogue
                if newfluxes.shape[0] == 0:
                    statement = 'No fluxes found in ' + cat.name + 'catalogue.'
                    print(statement)
                
                else:
                    newfluxes = newfluxes.set_index(pd.Index([int(self.id)]))
                    self.fluxes = pd.concat([self.fluxes, newfluxes], axis=1)

        # initialise fluxes
        else:
            
            data = cat.data

            # find source from id
            if hasattr(self, 'id'):
                
                self.fluxes = data[data[cat.columns[0]] == self.id]

                # update missing params
                if self.ra == None:
                    self.ra = data[data[cat.columns[0]] == self.id][cat.columns[2]].iloc[0]
                if self.dec == None:
                    self.dec = data[data[cat.columns[0]] == self.id][cat.columns[3]].iloc[0]

            # find source from position (closest in catalogue)
            else:

                ra_col = data[cat.columns[2]]
                dec_col = data[cat.columns[3]]
                sep_list = []
                for i in range (0, len(ra_col)):
                    sep_list.append(ang_sep(self.ra, ra_col[i], self.dec, dec_col[i]))
                sep_arr= np.array(sep_list)
                index = np.where(np.min(sep_arr) == sep_arr)[0][0]

                self.fluxes = data.iloc[index]
                self.id = data[data.columns[0]].iloc[index]

