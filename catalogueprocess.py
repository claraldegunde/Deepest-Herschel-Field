#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:06:52 2024

@author: claraaldegundemanteca
"""

import numpy as np

import pandas as pd

import sourceprocess as sp
from sourceprocess import from_m_to_F
from sourceprocess import from_hmsdms_to_deg

import astropy 
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
import math
#%%
cat_IRAC = sp.Catalogue('IRAC')
cat_IRAC_data = cat_IRAC.get_data('/Users/claraaldegundemanteca/Downloads/HerschelField /Chris_SPIREdarkfield/catalogues/IRACdark-matched_no_Nans.csv') # noNans.csv = nans are turned into 0


# cat_XID = sp.Catalogue('XID')
# cat_XID_data = cat_XID.get_data('/Users/claraaldegundemanteca/Downloads/HerschelField /Chris_SPIREdarkfield/catalogues/XIDmultiband.csv')
