#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 18:23:19 2024

@author: claraaldegundemanteca
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
#%% Function to generate datframe for CIGALE input/output

def generate_df (output_file):
    '''
    Generates a dataframe out of the CIGALE result.txt and observations.txt file

    Returns
    -------
    Pandas DataFrame
    '''
    output_data = pd.read_csv(output_file)
    #Create dataframe
    headers = output_data.columns[0]
    headers = headers.split(" ")
    headers_list = []
    for i in headers:
        # print(i)
        if i != '':
            headers_list.append(i)
       
    values_list = np.zeros([len(output_data), len(headers_list)])
    for i in range (0, len(output_data)): #scans through all sources
        values = output_data[output_data.columns[0]][i]
        values = values.split(" ")
        useful_values = []

        for j in values: # scans through values for the different variables
            # print(i)
            if j != '':
                useful_values.append(j)

        values_list [i, :] = useful_values
        
    return pd.DataFrame(data = values_list, columns = headers_list) #Dataframe created

#%% Create input dataframe

input_noSPIRE_visible_zCIGALE = generate_df('/Users/claraaldegundemanteca/Desktop/Herschel Field /CIGALE outputs/out_noSPIRE>10mJy_chi2<10/observations.txt')


#%% Create results dataframe (no SPIRE, only visible points, z from CIGALE) NO REQUIREMENT ON CHI2

#Create output dataframe
output_noSPIRE_visible_zCIGALE = generate_df ('/Users/claraaldegundemanteca/Desktop/Herschel Field /CIGALE outputs/out_noSPIRE_onlyvisible/results.txt')
output_noSPIRE_visible_zCIGALE = output_noSPIRE_visible_zCIGALE.rename(columns={"best.reduced_chi_square": "best_reduced_chi_square"})


#%% Create results dataframe (no SPIRE, only visible points, chi2<10, z from CIGALE)

# Impose condition on chi2 
chi2_less_than = 10

# Create new dataframe fro low reduced chi2 (less than 1 here but can change)
output_noSPIRE_visible_zCIGALE_chi2 = output_noSPIRE_visible_zCIGALE.query ('best_reduced_chi_square < %s and best_reduced_chi_square > 0' % (str(chi2_less_than)))

#Save list of IDs with small reduced chi2 and save dataframe with low reduced chi2
np.save('/Users/claraaldegundemanteca/Desktop/Herschel Field /Code/list_index_noSPIRE_red_chi2<%s.npy' % (str(chi2_less_than)), np.array(output_noSPIRE_visible_zCIGALE_chi2['id'], dtype=np.int32), allow_pickle = False)


#%% Create results dataframe with SPIRE, only visible points, z from CIGALE

#Create output dataframe
output_withSPIRE_visible_zCIGALE = generate_df ('/Users/claraaldegundemanteca/Desktop/Herschel Field /CIGALE outputs/out_withSPIRE_visible_zCIGALE/results_withSPIRE_visible_zCIGALE.txt')
output_withSPIRE_visible_zCIGALE = output_withSPIRE_visible_zCIGALE.rename(columns={"best.reduced_chi_square": "best_reduced_chi_square"})

# Impose condition on chi2 
chi2_less_than = 10

# Create new dataframe fro low reduced chi2 (less than 1 here but can change)
output_withSPIRE_visible_zCIGALE_chi2 = output_withSPIRE_visible_zCIGALE.query ('best_reduced_chi_square < %s and best_reduced_chi_square > 0' % (str(chi2_less_than)))

#Save list of IDs with small reduced chi2 and save dataframe with low reduced chi2
np.save('/Users/claraaldegundemanteca/Desktop/Herschel Field /Code/list_index_withSPIRE_red_chi2<%s.npy' % (str(chi2_less_than)), np.array(output_withSPIRE_visible_zCIGALE_chi2['id'], dtype=np.int32), allow_pickle = False)




#%% Create results dataframe with SPIRE > 30microJy, only visible points, z from CIGALE
chi2_less_than = 10
#Create output dataframe
output_withSPIRE_30mJy_visible_zCIGALE = generate_df ('/Users/claraaldegundemanteca/Desktop/Herschel Field /CIGALE outputs/out_withSPIRE>30mJy_chi2<10/results.txt')
output_withSPIRE_30mJy_visible_zCIGALE = output_withSPIRE_30mJy_visible_zCIGALE.rename(columns={"best.reduced_chi_square": "best_reduced_chi_square"})
#Chi2<10 already imposed here 


#Save list of IDs with small reduced chi2 and save dataframe with low reduced chi2
np.save('/Users/claraaldegundemanteca/Desktop/Herschel Field /Code/list_index_withSPIRE>30mJy_red_chi2<%s.npy' % (str(chi2_less_than)), np.array(output_withSPIRE_30mJy_visible_zCIGALE['id'], dtype=np.int32), allow_pickle = False)



#%% Create results dataframe noSPIRE (> 30mJy in withSPIRE), only visible points, z from CIGALE

#Create output dataframe
output_noSPIRE_30mJy_visible_zCIGALE = generate_df ('/Users/claraaldegundemanteca/Desktop/Herschel Field /CIGALE outputs/out_noSPIRE>30mJy_chi2<10/results.txt')
output_noSPIRE_30mJy_visible_zCIGALE = output_noSPIRE_30mJy_visible_zCIGALE.rename(columns={"best.reduced_chi_square": "best_reduced_chi_square"})
#Chi2<10 already imposed here 


#%% Create results dataframe with SPIRE > 10mJy, only visible points, z from CIGALE
chi2_less_than = 10
#Create output dataframe
output_withSPIRE_10mJy_visible_zCIGALE = generate_df ('/Users/claraaldegundemanteca/Desktop/Herschel Field /CIGALE outputs/out_withSPIRE>10mJy_chi2<10/results.txt')
output_withSPIRE_10mJy_visible_zCIGALE = output_withSPIRE_10mJy_visible_zCIGALE.rename(columns={"best.reduced_chi_square": "best_reduced_chi_square"})
#Chi2<10 already imposed here 


#Save list of IDs with small reduced chi2 and save dataframe with low reduced chi2
np.save('/Users/claraaldegundemanteca/Desktop/Herschel Field /Code/list_index_withSPIRE>10mJy_red_chi2<%s.npy' % (str(chi2_less_than)), np.array(output_withSPIRE_10mJy_visible_zCIGALE['id'], dtype=np.int32), allow_pickle = False)



#%% Create results dataframe noSPIRE (> 10microJy in withSPIRE), only visible points, z from CIGALE

#Create output dataframe
output_noSPIRE_10mJy_visible_zCIGALE = generate_df ('/Users/claraaldegundemanteca/Desktop/Herschel Field /CIGALE outputs/out_noSPIRE>10mJy_chi2<10/results.txt')
output_noSPIRE_10mJy_visible_zCIGALE = output_noSPIRE_10mJy_visible_zCIGALE.rename(columns={"best.reduced_chi_square": "best_reduced_chi_square"})
#Chi2<10 already imposed here 


#%% Check how chi2 changes when considering SPIRE and not (neglecting PSW < 30microJy)


# Set default font
plt.rcParams['font.serif'] = "Arial"
plt.rcParams['font.family'] = "Sans serif"
plt.rcParams['font.size'] = 18
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

def line(x, m, b):
    return m*x + b

x = np.linspace(0,6,100)
fig, axs = plt.subplots(1,1,figsize = (10,10))
t =  output_withSPIRE_10mJy_visible_zCIGALE['best.universe.redshift']
cs = axs.scatter(x = output_noSPIRE_10mJy_visible_zCIGALE['best_reduced_chi_square'] , y = output_withSPIRE_10mJy_visible_zCIGALE['best_reduced_chi_square'], c = t, cmap = 'rainbow', label= 'PSW > 10 mJy')
axs.plot(output_noSPIRE_30mJy_visible_zCIGALE['best_reduced_chi_square'] , output_withSPIRE_30mJy_visible_zCIGALE['best_reduced_chi_square'],  'o', markersize= 13, color = 'r', mfc='none', label= 'PSW > 30 mJy')
axs.plot(x, line(x, 1,0), label='$\chi^2_{SPIRE}$ = $\chi^2_{no SPIRE}$', c='#afe9ddff', linewidth = 0.7)
axs.plot(x, line(x, 1,0)+0.2*line(x, 1,0), label='$\pm 20\%$', c='#afe9ddff', linewidth = 0.7, linestyle='--')
axs.plot(x, line(x, 1,0)-0.2*line(x, 1,0),  c='#afe9ddff', linewidth = 0.7, linestyle='--')

axs.tick_params(labelsize = 12,direction='in',top=True,right=True,which='both')
axs.set_xlabel('Reduced $\chi^2$ without SPIRE', fontsize = 18)
axs.set_ylabel('Reduced $\chi^2$  with SPIRE',  fontsize = 18)
axs.xaxis.set_minor_locator(MultipleLocator(0.5))
axs.yaxis.set_minor_locator(MultipleLocator(0.5))
axs.grid(alpha=0.3)
fig.colorbar(cs, label = 'z')
axs.legend(facecolor = 'k', framealpha = 0.1, loc='lower right')
axs.spines['bottom'].set_color('white')
axs.spines['top'].set_color('white')
axs.spines['right'].set_color('white')
axs.spines['left'].set_color('white')

fig.savefig('chi_relation_ok.png', dpi = 300, transparent = True)


# no_bins = 30
# fig, axs = plt.subplots(1,1,figsize = (8,8))
# axs.hist(output_noSPIRE_30mJy_visible_zCIGALE['best_reduced_chi_square'],bins=no_bins, histtype = 'step', label='Without SPIRE',  facecolor='red')
# axs.hist(output_withSPIRE_30mJy_visible_zCIGALE['best_reduced_chi_square'],bins=no_bins,histtype = 'step', label= 'With SPIRE', facecolor='black')
# axs.set_xlabel('Red $\chi^2$', fontsize = 16)
# axs.set_ylabel('Frequency',  fontsize = 16)
# axs.xaxis.set_minor_locator(MultipleLocator(50))
# axs.yaxis.set_minor_locator(MultipleLocator(500))
# axs.set_title('Comparing $\chi^2$ with and without SPIRE points ',fontsize = 20)
# axs.grid(alpha=0.5)
# axs.legend()



#%% Distributions to study the field 

# Set default font
plt.rcParams['font.serif'] = "Arial"
plt.rcParams['font.family'] = "Sans serif"
plt.rcParams['font.size'] = 30
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'


#Z distribution
no_bins = 20
fig, axs = plt.subplots(4,1, figsize=(16,20))
fig.tight_layout(pad=1.5)

axs[0].hist(output_noSPIRE_visible_zCIGALE_chi2 ['best.universe.redshift'],bins=np.linspace(0,5, no_bins),  color ='#97FEED',  edgecolor ='k',  label= 'No SPIRE', alpha = 0.7)

axs[1].hist(output_withSPIRE_10mJy_visible_zCIGALE['best.universe.redshift'],bins=np.linspace(0,5, no_bins),  color ='#9EDE73', edgecolor ='k', label= 'PSW > 10 mJy', alpha = 0.7)
axs[1].hist(output_withSPIRE_30mJy_visible_zCIGALE['best.universe.redshift'],bins=np.linspace(0,5, no_bins), color = '#FFC600', edgecolor = 'k', label= 'PSW > 30 mJy',  alpha = 0.7)

axs[2].hist(np.log10(output_withSPIRE_10mJy_visible_zCIGALE['bayes.stellar.m_star']),bins=np.linspace(9,12, no_bins), color ='#9EDE73', edgecolor ='k', label= 'PSW > 10 mJy', alpha = 0.7)
axs[2].hist(np.log10(output_withSPIRE_30mJy_visible_zCIGALE['bayes.stellar.m_star']),bins=np.linspace(9,12, no_bins), color = '#FFC600', edgecolor = 'k', label= 'PSW > 30 mJy',  alpha = 0.7)

axs[3].hist(np.log10(output_withSPIRE_10mJy_visible_zCIGALE['bayes.sfh.sfr']),bins=np.linspace(-1,4, no_bins),  color ='#9EDE73', edgecolor ='k', label= 'PSW > 10 mJy', alpha = 0.7)
axs[3].hist(np.log10(output_withSPIRE_30mJy_visible_zCIGALE['bayes.sfh.sfr']),bins=np.linspace(-1,4, no_bins), color = '#FFC600', edgecolor = 'k', label= 'PSW > 30 mJy',  alpha = 0.7)

vals_10, bin_edges_10 = np.histogram(output_withSPIRE_10mJy_visible_zCIGALE['best.universe.redshift'], bins = no_bins, range = (0, 5))
vals_30, bin_edges_30 = np.histogram(output_withSPIRE_30mJy_visible_zCIGALE['best.universe.redshift'], bins = no_bins, range = (0, 5))

vals_mass, bin_edges_mass = np.histogram(np.log10(output_withSPIRE_10mJy_visible_zCIGALE['bayes.stellar.m_star']), bins=np.linspace(9,12, no_bins))
bin_centre_mass = [(bin_edges_mass[i] + bin_edges_mass[i+1])/2 for i in range(0, len(bin_edges_mass)-1)] 
z_mass_list = []
for i in range(1,len(bin_edges_mass)):
    df = output_withSPIRE_10mJy_visible_zCIGALE[output_withSPIRE_10mJy_visible_zCIGALE['bayes.stellar.m_star'].between(10**(bin_edges_mass[i-1]), 10**(bin_edges_mass[i]))]
    print(len(df))
    # print(df['best.universe.redshift'])
    z_mass_list.append( np.mean(df['best.universe.redshift'] ))

vals_sfr, bin_edges_sfr = np.histogram(np.log10(output_withSPIRE_10mJy_visible_zCIGALE['bayes.sfh.sfr']), bins=np.linspace(-1,4, no_bins))
bin_centre_sfr = [(bin_edges_sfr[i] + bin_edges_sfr[i+1])/2 for i in range(0, len(bin_edges_sfr)-1)] 
z_sfr_list = []
for i in range(1,len(bin_edges_sfr)):
    df = output_withSPIRE_10mJy_visible_zCIGALE[output_withSPIRE_10mJy_visible_zCIGALE['bayes.sfh.sfr'].between(10**(bin_edges_sfr[i-1]), 10**(bin_edges_sfr[i]))]
    # print(len(df))
    z_sfr_list.append( np.mean(df['best.universe.redshift'] ))


axs[0].set_ylabel('Frequency')
axs[0].tick_params(labelsize = 30,direction='in',top=True,right=True,which='both')
axs[0].xaxis.set_minor_locator(MultipleLocator(0.5))
axs[0].yaxis.set_minor_locator(MultipleLocator(1000))
# axs[0].grid(alpha=0.3, which = 'both')
axs[0].legend(facecolor = 'k', framealpha = 0.1)
axs[0].spines['bottom'].set_color('white')
axs[0].spines['top'].set_color('white')
axs[0].spines['right'].set_color('white')
axs[0].spines['left'].set_color('white')
axs[0].set_xlabel('z')

axs[1].set_xlabel('z')
axs[1].set_ylabel('Frequency')
axs[1].tick_params(labelsize = 30,direction='in',top=True,right=True,which='both')
axs[1].xaxis.set_minor_locator(MultipleLocator(0.5))
axs[1].yaxis.set_minor_locator(MultipleLocator(2.5))
# axs[1].grid(alpha=0.3, which = 'both')
axs[1].legend(facecolor = 'k', framealpha = 0.1)
axs[1].spines['bottom'].set_color('white')
axs[1].spines['top'].set_color('white')
axs[1].spines['right'].set_color('white')
axs[1].spines['left'].set_color('white')

axs[2].set_xlabel('z')
axs[2].set_ylabel('Frequency')
axs[2].tick_params(labelsize = 30,direction='in',top=True,right=True,which='both')
axs[2].xaxis.set_minor_locator(MultipleLocator(0.5))
axs[2].yaxis.set_minor_locator(MultipleLocator(2.5))
# axs[2].grid(alpha=0.3, which = 'both')
axs[2].legend(facecolor = 'k', framealpha = 0.1,  loc='upper left')
axs[2].spines['bottom'].set_color('white')
axs[2].spines['top'].set_color('white')
axs[2].spines['right'].set_color('white')
axs[2].spines['left'].set_color('white')
axs[2].set_xlabel('log stellar mass  $(M_{\odot})$')
axs_z = axs[2].twinx() 
axs_z.plot(bin_centre_mass, z_mass_list,'+',  markersize=10, label='z', color = 'white')
axs_z.legend(facecolor = 'k', framealpha = 0.1, loc='center left')
axs_z.set_ylabel('z')
axs_z.spines['bottom'].set_color('white')
axs_z.spines['top'].set_color('white')
axs_z.spines['right'].set_color('white')
axs_z.spines['left'].set_color('white')
axs_z.yaxis.set_major_locator(MultipleLocator(1))


axs[3].set_ylabel('Frequency')
axs[3].tick_params(labelsize = 30,direction='in',top=True,right=True,which='both')
axs[3].xaxis.set_minor_locator(MultipleLocator(0.5))
axs[3].yaxis.set_minor_locator(MultipleLocator(1))
# axs[3].grid(alpha=0.3, which = 'both')
axs[3].legend(facecolor = 'k', framealpha = 0.1, loc='upper left')
axs[3].spines['bottom'].set_color('white')
axs[3].spines['top'].set_color('white')
axs[3].spines['right'].set_color('white')
axs[3].spines['left'].set_color('white')
axs[3].set_xlabel('log SFR $(M_{\odot}/ yr)$')
axs_z = axs[3].twinx() 
axs_z.plot(bin_centre_sfr, z_sfr_list,'+',  markersize=10, label='z',color = 'white')
axs_z.legend(facecolor = 'k', framealpha = 0.1, loc='center left')
axs_z.set_ylabel('z')
axs_z.spines['bottom'].set_color('white')
axs_z.spines['top'].set_color('white')
axs_z.spines['right'].set_color('white')
axs_z.spines['left'].set_color('white')
axs_z.yaxis.set_major_locator(MultipleLocator(1))

fig.savefig('z_distribution.png', dpi = 300, transparent = True)

#%%


vals_10, bin_edges_10 = np.histogram(output_withSPIRE_10mJy_visible_zCIGALE['best.universe.redshift'], bins = no_bins, range = (0, 5))
vals_30, bin_edges_30 = np.histogram(output_withSPIRE_30mJy_visible_zCIGALE['best.universe.redshift'], bins = no_bins, range = (0, 5))

vals_mass, bin_edges_mass = np.histogram(np.log10(output_withSPIRE_10mJy_visible_zCIGALE['bayes.stellar.m_star']), bins = 20)
bin_centre_mass = [(bin_edges_mass[i] + bin_edges_mass[i+1])/2 for i in range(0, len(bin_edges_mass)-1)] 
z_mass_list = []
for i in range(1,len(bin_edges_mass)):
    df = output_withSPIRE_10mJy_visible_zCIGALE[output_withSPIRE_10mJy_visible_zCIGALE['bayes.stellar.m_star'].between(10**(bin_edges_mass[i-1]), 10**(bin_edges_mass[i]))]
    print(len(df))
    z_mass_list.append( np.mean(df['best.universe.redshift'] ))
vals_sfr, bin_edges_sfr = np.histogram(np.log10(output_withSPIRE_10mJy_visible_zCIGALE['bayes.sfh.sfr']), bins = 20)
bin_centre_sfr = [(bin_edges_sfr[i] + bin_edges_sfr[i+1])/2 for i in range(0, len(bin_edges_sfr)-1)] 
z_sfr_list = []
for i in range(1,len(bin_edges_sfr)):
    df = output_withSPIRE_10mJy_visible_zCIGALE[output_withSPIRE_10mJy_visible_zCIGALE['bayes.sfh.sfr'].between(10**(bin_edges_sfr[i-1]), 10**(bin_edges_sfr[i]))]
    print(len(df))
    z_sfr_list.append( np.mean(df['best.universe.redshift'] ))
    

#%% Check bayes and best redshift are not too different 


# No SPIRE
x = np.linspace(0,6,100)
fig, axs = plt.subplots(1,1,figsize = (8,8))
axs.errorbar(output_noSPIRE_visible_zCIGALE_reduced['best.universe.redshift'], output_noSPIRE_visible_zCIGALE_reduced['bayes.universe.redshift'], yerr = output_noSPIRE_visible_zCIGALE_reduced['bayes.universe.redshift_err'], fmt='x', c='#D55E00')
axs.plot(x, line(x, 1,0), label='$z_{Bayes} = z_{best}$', c='k')
axs.set_xlabel('$z_{best}$', fontsize = 16)
axs.set_ylabel('$z_{Bayes}$',  fontsize = 16)
axs.tick_params(labelsize = 12,direction='in',top=True,right=True,which='both')
axs.xaxis.set_minor_locator(MultipleLocator(0.5))
axs.yaxis.set_minor_locator(MultipleLocator(0.5))
axs.set_title('Checking correlation between $z_{Bayes}$ and $z_{best}$',fontsize = 20)
axs.grid(alpha=0.5)
axs.legend()

# With SPIRE
x = np.linspace(0,6,100)
fig, axs = plt.subplots(1,1,figsize = (8,8))
axs.errorbar(output_withSPIRE_visible_zCIGALE_reduced['best.universe.redshift'], output_withSPIRE_visible_zCIGALE_reduced['bayes.universe.redshift'], yerr = output_withSPIRE_visible_zCIGALE_reduced['bayes.universe.redshift_err'], fmt='x', c='#D55E00')
axs.plot(x, line(x, 1,0), label='$z_{Bayes} = z_{best}$', c='k')
axs.set_xlabel('$z_{best}$', fontsize = 16)
axs.set_ylabel('$z_{Bayes}$',  fontsize = 16)
axs.tick_params(labelsize = 12,direction='in',top=True,right=True,which='both')
axs.xaxis.set_minor_locator(MultipleLocator(0.5))
axs.yaxis.set_minor_locator(MultipleLocator(0.5))
axs.set_title('Checking correlation between $z_{Bayes}$ and $z_{best}$',fontsize = 20)
axs.grid(alpha=0.5)
axs.legend()

'''
If no big difference between the two values we don't ned to discard any sources
'''


#%% chi2 against redshift 

# chi2 = np.array(chi2)
# redshift = np.array(chi2)

plt.figure()
plt.scatter(output_withSPIRE_visible_zCIGALE_reduced['best.universe.redshift'], output_withSPIRE_visible_zCIGALE_reduced['best_reduced_chi_square'],color='k', alpha = 0.7)
plt.grid(alpha=0.7)
plt.ylabel('$\chi^2$')
plt.xlabel('z')
plt.title('$\chi^2$ vs z ')

#%% Luminosity against z 

x = np.linspace(0,6,100)
fig, axs = plt.subplots(1,1,figsize = (8,8))
axs.plot(output_withSPIRE_visible_zCIGALE_reduced['best.universe.redshift'], output_withSPIRE_visible_zCIGALE_reduced['best.stellar.lum'], 'o', c='#D55E00')
axs.set_yscale('log')
axs.set_xlabel('$z_{best}$', fontsize = 16)
axs.set_ylabel('$Luminosity$',  fontsize = 16)
# axs.tick_params(labelsize = 12,direction='in',top=True,right=True,which='both')
# axs.xaxis.set_minor_locator(MultipleLocator(0.5))
# axs.yaxis.set_minor_locator(MultipleLocator(0.5))
axs.set_title('Checking correlation between $z_{Bayes}$ and $z_{best} (with SPIRE)$',fontsize = 16)
axs.grid(alpha=0.5)



#%% SFR against mass for points with PSW > 10 

# Set default font
plt.rcParams['font.serif'] = "Arial"
plt.rcParams['font.family'] = "Sans serif"
plt.rcParams['font.size'] = 14
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'


output_withSPIRE_10mJy_visible_zCIGALE = output_withSPIRE_10mJy_visible_zCIGALE.rename(columns={"best.universe.redshift": "best_universe_redshift"})
output_withSPIRE_30mJy_visible_zCIGALE = output_withSPIRE_30mJy_visible_zCIGALE.rename(columns={"best.universe.redshift": "best_universe_redshift"})

bin_1 = output_withSPIRE_10mJy_visible_zCIGALE.query('best_universe_redshift < 0.25 and best_reduced_chi_square < 5')
bin_1_30 = output_withSPIRE_30mJy_visible_zCIGALE.query('best_universe_redshift < 0.25 and best_reduced_chi_square < 5')

bin_2 = output_withSPIRE_10mJy_visible_zCIGALE.query('best_universe_redshift > 0.25 and best_universe_redshift < 0.5 and best_reduced_chi_square < 5')
bin_3 = output_withSPIRE_10mJy_visible_zCIGALE.query('best_universe_redshift > 0.5 and best_universe_redshift < 0.75 and best_reduced_chi_square < 5')
bin_4 = output_withSPIRE_10mJy_visible_zCIGALE.query('best_universe_redshift > 0.75 and best_universe_redshift < 1 and best_reduced_chi_square < 5')
bin_5 = output_withSPIRE_10mJy_visible_zCIGALE.query('best_universe_redshift > 1 and best_universe_redshift < 1.25 and best_reduced_chi_square < 5')
bin_6 = output_withSPIRE_10mJy_visible_zCIGALE.query('best_universe_redshift > 1.25 and best_universe_redshift < 1.5 and best_reduced_chi_square < 5')



bayes_sfr_1 = bin_1['bayes.sfh.sfr']
bayes_sft_err_1 = bin_1['bayes.sfh.sfr_err']
bayes_M_1 = bin_1['bayes.stellar.m_star']
bayes_M_err_1 = bin_1['bayes.stellar.m_star_err']

bayes_sfr_1_30 = bin_1_30['bayes.sfh.sfr']
bayes_sft_err_1_30 = bin_1_30['bayes.sfh.sfr_err']
bayes_M_1_30 = bin_1_30['bayes.stellar.m_star']
bayes_M_err_1_30 = bin_1_30['bayes.stellar.m_star_err']

bayes_sfr_2 = bin_2['bayes.sfh.sfr']
bayes_sft_err_2 = bin_2['bayes.sfh.sfr_err']
bayes_M_2 = bin_2['bayes.stellar.m_star']
bayes_M_err_2 = bin_2['bayes.stellar.m_star_err']

bayes_sfr_3 = bin_3['bayes.sfh.sfr']
bayes_sft_err_3 = bin_3['bayes.sfh.sfr_err']
bayes_M_3 = bin_3['bayes.stellar.m_star']
bayes_M_err_3 = bin_3['bayes.stellar.m_star_err']

bayes_sfr_4 = bin_4['bayes.sfh.sfr']
bayes_sft_err_4 = bin_4['bayes.sfh.sfr_err']
bayes_M_4 = bin_4['bayes.stellar.m_star']
bayes_M_err_4 = bin_4['bayes.stellar.m_star_err']


bayes_sfr_5 = bin_5['bayes.sfh.sfr']
bayes_sft_err_5 = bin_5['bayes.sfh.sfr_err']
bayes_M_5 = bin_5['bayes.stellar.m_star']
bayes_M_err_5 = bin_5['bayes.stellar.m_star_err']


bayes_sfr_6 = bin_6['bayes.sfh.sfr']
bayes_sft_err_6 = bin_6['bayes.sfh.sfr_err']
bayes_M_6 = bin_6['bayes.stellar.m_star']
bayes_M_err_6 = bin_6['bayes.stellar.m_star_err']


best_sfr_1 = bin_1['best.sfh.sfr']
best_M_1 = bin_1['best.stellar.m_star']

best_sfr_2 = bin_2['best.sfh.sfr']
best_M_2 = bin_2['best.stellar.m_star']

best_sfr_3 = bin_3['best.sfh.sfr']
best_M_3 = bin_3['best.stellar.m_star']

best_sfr_4 = bin_4['best.sfh.sfr']
best_M_4 = bin_4['best.stellar.m_star']

best_sfr_5 = bin_5['best.sfh.sfr']
best_M_5 = bin_5['best.stellar.m_star']

best_sfr_6 = bin_6['best.sfh.sfr']
best_M_6 = bin_6['best.stellar.m_star']

# fig, axs = plt.subplots(1,1,figsize = (8,8))
# axs.set_xscale('log')
# axs.set_yscale('log')
# axs.errorbar(bayes_sfr_1 , bayes_M_1, xerr = bayes_sft_err_1 , yerr = bayes_M_err_1, fmt = 'x', label = 'z < 1')
# axs.errorbar(bayes_sfr_2 , bayes_M_2, xerr = bayes_sft_err_2 , yerr = bayes_M_err_2, fmt = 'x', label = '1 < z < 2 ')
# axs.errorbar(bayes_sfr_3 , bayes_M_3, xerr = bayes_sft_err_3 , yerr = bayes_M_err_3, fmt = 'x', label = '2 < z < 3 ')
# axs.errorbar(bayes_sfr_4 , bayes_M_4, xerr = bayes_sft_err_4 , yerr = bayes_M_err_4, fmt = 'x', label = '3 < z < 5 ')
# axs.errorbar(bayes_sfr_5 , bayes_M_5, xerr = bayes_sft_err_5 , yerr = bayes_M_err_5, fmt = 'x', label = '3 < z < 5 ')
# axs.errorbar(bayes_sfr_6 , bayes_M_6, xerr = bayes_sft_err_6 , yerr = bayes_M_err_6, fmt = 'x', label = '3 < z < 5 ')
# axs.set_xlabel('M')
# axs.set_ylabel('SFR')
# axs.spines['bottom'].set_color('white')
# axs.spines['top'].set_color('white')
# axs.spines['right'].set_color('white')
# axs.spines['left'].set_color('white')
# axs.grid(alpha=0.3)
# fig.savefig('SFR_mass_bayes.png', dpi = 300, transparent = True)

#Points from Speagle 
from scipy.optimize import curve_fit

x_0 = np.array(([9.7, 11.1]))
y_0 = np.array(([-0.25, 0.4]))
coef_0,_ = curve_fit(line, x_0, y_0)

x_025 = np.array(([9.7, 11.1]))
y_025 = np.array(([0.2, 0.8]))
coef_025,_ = curve_fit(line, x_025, y_025)


x_range = np.linspace(9, 11, 10000)
fig, axs = plt.subplots(1,1,figsize = (8,8))
axs.plot(np.log10(bayes_M_1), np.log10(bayes_sfr_1), 'x', label = '0 < z < 0.25')
axs.plot(np.log10(bayes_M_1_30), np.log10(bayes_sfr_1_30), 'o',  mfc='none', label = '0 < z < 0.25 PSW > 30 mJy')

# axs.plot(x_025, y_025)
axs.plot(x_range, line(x_range, coef_025[0], coef_025[1]), label =' z = 0.25')
axs.plot(x_range, line(x_range, coef_0[0], coef_0[1]), label =' z = 0')
# axs.plot( best_M_2,best_sfr_2 , 'x', label = '0.25 < z < 0.50')
# axs.plot( best_M_3,best_sfr_3 , 'x', label = '0.50 < z < 0.75')
# axs.plot(best_M_4, best_sfr_4 , 'x', label = '0.75 < z < 1.00')
# axs.plot(best_M_5, best_sfr_5 , 'x', label = '1.00 < z < 1.25')
# axs.plot(best_M_6, best_sfr_6 , 'x', label = '1.00 < z < 1.25')

axs.set_xlabel('log stellar mass  $(M_{\odot})$')
axs.set_ylabel('log SFR $(M_{\odot}/ yr)$')
axs.spines['bottom'].set_color('white')
axs.spines['top'].set_color('white')
axs.spines['right'].set_color('white')
axs.spines['left'].set_color('white')
axs.grid(alpha=0.3)
axs.legend(facecolor = 'k', framealpha = 0.1)
fig.savefig('SFR_mass_best.png', dpi = 300, transparent = True)


#%%

fig, axs = plt.subplots(1,1,figsize = (8,8))
axs.set_yscale('log')
axs.plot(output_withSPIRE_10mJy_visible_zCIGALE['best.universe.redshift'], output_withSPIRE_10mJy_visible_zCIGALE['best.sfh.sfr'], 'x')
fig.savefig('SFR_redshift_best.png', dpi = 300, transparent = True)

#%%
plt.rcParams['font.serif'] = "Arial"
plt.rcParams['font.family'] = "Sans serif"
plt.rcParams['font.size'] = 16
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

from scipy.optimize import curve_fit

x_top = np.array([10**7, 10**(7.7), 10**(8.2)])
y_top = np.array([10**(-0.2), 10**(0.5), 10])
coef_top,_ = curve_fit(line, x_top, y_top)

x_bottom = np.array([10**(8.7), 10**(9.7), 10**(10.7)])
y_bottom = np.array([ 10, 10**2, 10**3])
coef_bottom,_ = curve_fit(line, x_bottom, y_bottom)


x_range = np.linspace(10**7, 10**12, 1000)
fig, axs = plt.subplots(1,1,figsize = (8,8))
axs.set_yscale('log')
axs.set_xscale('log')
axs.set_xlabel('Stellar mass  $(M_{\odot})$')
axs.set_ylabel('SFR $(M_{\odot}/ yr)$')
axs.plot(output_withSPIRE_10mJy_visible_zCIGALE['best.stellar.m_star'][2:], output_withSPIRE_10mJy_visible_zCIGALE['best.sfh.sfr'][2:], 'x', label= 'PSW > 10 mJy')
axs.plot(output_withSPIRE_30mJy_visible_zCIGALE['best.stellar.m_star'][2:], output_withSPIRE_30mJy_visible_zCIGALE['best.sfh.sfr'][2:], 'o', mfc='none', label= 'PSW > 30 mJy')
axs.plot(x_range,line(x_range, *coef_top))
axs.plot(x_range,line(x_range, *coef_bottom))
axs.legend()

fig.savefig('SFR_redshift_best.png', dpi = 300, transparent = True)

#%%

plt.rcParams['font.serif'] = "Arial"
plt.rcParams['font.family'] = "Sans serif"
plt.rcParams['font.size'] = 16
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

fig, axs = plt.subplots(1,2,figsize = (20,8))
axs[0].hist(np.log10(output_withSPIRE_10mJy_visible_zCIGALE['bayes.stellar.m_star']),bins=10, density = True, color ='#e9afafff', edgecolor ='k', label= 'PSW > 10 mJy', alpha = 1)
axs[1].hist(np.log10(output_withSPIRE_10mJy_visible_zCIGALE['bayes.sfh.sfr']),bins=10, density = True,  color ='#e9afafff', edgecolor ='k', label= 'PSW > 10 mJy', alpha = 1)


