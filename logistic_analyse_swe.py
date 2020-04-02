# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:59:42 2020

@author: Johan Odelius
Luleå University of Technology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

#%% Read sweden
url = 'https://www.arcgis.com/sharing/rest/content/items/b5e7488e117749c19881cce45db13f7e/data'
df_cases = pd.read_excel(url,sheet_name=0)
df_cases.index = pd.to_datetime(df_cases['Statistikdatum'])
last_update = pd.to_datetime(df_cases['Statistikdatum'][-1])
del df_cases['Statistikdatum']

df_deaths = pd.read_excel(url,sheet_name=1)
df_deaths.iloc[df_deaths['Datum_avliden'].values=='Uppgift saknas',0]=''
df_deaths.index = pd.to_datetime(df_deaths['Datum_avliden'])
del df_deaths['Datum_avliden']

df_iva = pd.read_excel(url,sheet_name=2)
df_iva.index = pd.to_datetime(df_iva['Datum_vårdstart'])
df_iva.loc[:,'Antal_intensivvårdade'].fillna(0,inplace=True)  
del df_iva['Datum_vårdstart']



#%% Plot selected countries
fig,axes = plt.subplots(nrows=2,ncols=2)
df_cases.loc[:,'Totalt_antal_fall'].cumsum().plot(ax=axes[0,0],title='Antal fall')
df_deaths.loc[:,'Antal_avlidna'].cumsum().plot(ax=axes[1,0],title='Antal avlidna')
df_cases.loc[:,'Totalt_antal_fall'].plot(ax=axes[0,1],title='Antal fall per dag')
df_deaths.loc[:,'Antal_avlidna'].plot(ax=axes[1,1],title='Antal avlidna per dag')

    

#%% SUBFUNCTIONS
def logistic_mdl(x,a,b,c):
    #a refers to the infection speed
    #b is the day with the maximum infections occurred
    #c is the total number of recorded infected people at the infection’s end
    return c/(1+np.exp(-(x-b)/a))
    
def prepare(df, col, xhat=np.arange(0,100)):
    dt=df.index-df.index[0]
    x0 = dt.days.values
    
    if isinstance(col,str):
        ys = df.loc[:,col].cumsum()
        dys = df.loc[:,col]
    else:
        ys = df.iloc[:,col].cumsum()
        dys = df.iloc[:,col]
    
    # set x=0 when first case is reported
    #i = np.where(np.logical_and(ys.values>0,np.isfinite(x0)))[0]
    i = np.where(np.logical_and(ys.values>0,df.index<last_update))[0]    
    y = ys.values[i]
    dy = dys.values[i]
    
    x = x0[i]-x0[i[0]]
    t = df.index.values[i]
    
    that = t
    while len(that)<len(xhat):
        that = np.append(that,that[-1]+np.timedelta64(1,'D'))
    
    return {'x':x, 'y':y, 't':t, 'xhat':xhat, 'that':that, 'dy':dy}    

#%% 
def fit(df, data_label=None, cols=[0], p0=None):
    # Init plot
    Nc = len(cols)
    nax = (2,Nc)    
    fig, axes = plt.subplots(nrows=nax[0], ncols=nax[1])
         
    output = {}     
        
    for n,col in enumerate(cols):
        # Data preperation for fit
        data = prepare(df,col)
        
        # Plot data
        plt.subplot(nax[0],nax[1],n+1)
        plt.plot(data['t'], data['y'], '-', label='data')
        plt.ylabel(data_label)
       
        try:
            # Logistic function fit (i.e. cdf of logistic distribution)
            popt, pcov = curve_fit(logistic_mdl, data['x'], data['y'], p0=p0)
            data['yhat'] = logistic_mdl(data['xhat'], *popt)
            
            plt.plot(data['that'], data['yhat'], 'r--', 
                     label='logistic fit (speed=%5.2f)'%popt[0])
        
        except:
            print('Error fitting'+col)
        
        plt.legend()        
        
        output[col] = {
                'data':data,
                'mdl':{
                        'fun':logistic_mdl,
                        'p':popt
                        }
                }
    
    # Ploting per day and logistic pdf
    for n,col in enumerate(cols):
        data = output[col]['data']
        popt = output[col]['mdl']['p']
        
        # Plot cases per day
        plt.subplot(nax[0],nax[1],n+1+Nc)
        plt.bar(data['t'], data['dy'], label='data')
        plt.ylabel(data_label+' per dag')
        
        # Plot logistic pdf
        dyhat = popt[2]*stats.logistic.pdf(data['xhat'],loc=popt[1],scale=popt[0])
        plt.plot(data['that'], dyhat, 'r--', label='fit (logistic pdf)')
        
        plt.legend()
    
    return output
         
#%%
m = fit(df_cases, data_label='Fall', cols=[0], p0=[5,50,10000])
m = fit(df_deaths, data_label='Avlidna', cols=[0], p0=[5,20,2000])
m = fit(df_iva, data_label='IVA', cols=[0])
#plt.savefig('Deathplot.png')
