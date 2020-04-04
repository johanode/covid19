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
import os

#%% Read sweden
url = 'https://www.arcgis.com/sharing/rest/content/items/b5e7488e117749c19881cce45db13f7e/data'
xl = pd.ExcelFile(url)

print(xl.sheet_names[-1])

# Save local
i_would_like_to_save_data_to_local_file = 'no' #(yes/no)
filepath = 'data/'
if i_would_like_to_save_data_to_local_file.lower() == 'yes':
    filename = xl.sheet_names[-1]+'.xlsx'
    file_exist = os.path.isfile(filepath+filename)
    if file_exist:
        do_write=input('File "%s" exist, overwrite (yes/no)?' % filename)
    
    if not file_exist or 'y' in do_write:
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        print('Writing '+filename)
        with pd.ExcelWriter(filepath+filename) as writer:
            for sheet in xl.sheet_names:
                df = xl.parse(sheet)
                df.to_excel(writer, sheet_name=sheet, index=False)
                

df_cases = xl.parse('Antal per dag region') #pd.read_excel(url,sheet_name=0)
df_cases.index = pd.to_datetime(df_cases['Statistikdatum'])
last_update = pd.to_datetime(df_cases['Statistikdatum'][-1])
del df_cases['Statistikdatum']

df_deaths = xl.parse('Antal avlidna per dag') #pd.read_excel(url,sheet_name=1)
df_deaths.iloc[df_deaths['Datum_avliden'].values=='Uppgift saknas',0]=''
df_deaths.index = pd.to_datetime(df_deaths['Datum_avliden'])
del df_deaths['Datum_avliden']

df_iva = xl.parse('Antal intensivvårdade per dag') #pd.read_excel(url,sheet_name=2)
df_iva.index = pd.to_datetime(df_iva['Datum_vårdstart'])
df_iva.loc[:,'Antal_intensivvårdade'].fillna(0,inplace=True)  
del df_iva['Datum_vårdstart']

xl.close()

#%% Merge to one dataframe
df_cases.index.name = 'Datum'
df_deaths.index.name = 'Datum'
df_iva.index.name = 'Datum'
df = df_cases
df[df_deaths.columns[0]] = df_deaths.iloc[:,0]
df[df_iva.columns[0]] = df_iva.iloc[:,0]

# Replace missing values with zero
df.fillna(0,inplace=True)  

# set x=0 when first case is reported
# and exclude last date
valid_cases ={col: np.logical_and(df[col].cumsum().values>0,df.index<last_update) for col in df.keys()}

#%% Plot selected columns
fig,axes = plt.subplots(nrows=2,ncols=1)
cols = ['Totalt_antal_fall','Antal_avlidna','Antal_intensivvårdade']
df.loc[:,cols].cumsum().plot(ax=axes[0],title='Antal')
df.loc[:,cols].plot(ax=axes[1],title='Antal per dag')

    

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
    
    # set x=0 when first case is reported and exclude last date    
    i = np.where(valid_cases[col])[0]
    y = ys.values[i]
    dy = dys.values[i]
    
    x = x0[i]-x0[i[0]]
    t = df.index.values[i]
    
    that = t
    while len(that)<len(xhat):
        that = np.append(that,that[-1]+np.timedelta64(1,'D'))
    
    return {
            'data':{'x':x, 'y':y, 't':t, 'dy':dy}, 
            'fit':{'x':xhat, 't':that}
            }    

def fit(df, data_label=None, cols=[0], p0=None):    
    output = {}               
    for n,col in enumerate(cols):
        # Data preperation for fit
        output[col] = prepare(df,col)
        
        try:
            # Logistic function fit (i.e. cdf of logistic distribution)
            popt, pcov = curve_fit(logistic_mdl, output[col]['data']['x'], output[col]['data']['y'], p0=p0)
            output[col]['fit']['y'] = logistic_mdl(output[col]['fit']['x'], *popt) 
            output[col]['mdl'] = {'fun':logistic_mdl, 'p':popt}                        
            
            # Per day and logistic pdf          
            # logistic pdf                    
            output[col]['fit']['dy'] = popt[2]*stats.logistic.pdf(output[col]['fit']['x'],loc=popt[1],scale=popt[0])
        except:
            output[col]['fit']['y'] = []
            output[col]['fit']['dy'] = []
            output[col]['mdl'] = {'fun':logistic_mdl, 'p':[]}
            print('Error fitting'+col)                        
                
    return output
  
#%% Fit logistic model and save to dataframe
m_1 = fit(df, data_label='Antal fall', cols=df.keys(), p0=[5,50,10000]) #['Totalt_antal_fall']
m_2 = fit(df, data_label='Antal', cols=['Antal_avlidna','Antal_intensivvårdade'], p0=[4.55346113e+00, 7.73342047e+01, 8.82173825e+03]) #[4,50,1000]
m = {**m_1,**m_2} 

#%% plot
cols = ['Totalt_antal_fall']
# Init plot
fig,axes = plt.subplots(nrows=2,ncols=1)
colors = ['C0','C1','C2','C3','C4','C5','C6']
# Plot data  
ax11 = plt.subplot(211)  
for col in cols: 
    plt.plot(m[col]['data']['t'], m[col]['data']['y'], colors[0]+':.', label=col)         
    plt.plot(m[col]['fit']['t'], m[col]['fit']['y'], colors[1]+'-',label=col+' logistic fit')    
    plt.ylabel('Antal')
    plt.legend(loc=2)
ax12 = ax11.twinx()
for n,col in enumerate(['Antal_avlidna','Antal_intensivvårdade']): 
    plt.plot(m[col]['data']['t'], m[col]['data']['y'], colors[2*n+2]+':.', label=col) 
    plt.plot(m[col]['fit']['t'], m[col]['fit']['y'], colors[2*n+3]+'-',label=col+' logistic fit')  
    plt.ylabel('Antal')
    plt.legend(loc=4)

    
ax21 = plt.subplot(212)  
for col in cols: 
    plt.plot(m[col]['data']['t'], m[col]['data']['dy'], colors[0]+':.', label=col)         
    plt.plot(m[col]['fit']['t'], m[col]['fit']['dy'], colors[1]+'-',label=col+' logistic fit')          
    plt.ylabel('Antal per dag')
    plt.legend(loc=2)
ax22 = ax21.twinx()
for n,col in enumerate(['Antal_avlidna','Antal_intensivvårdade']): 
    plt.plot(m[col]['data']['t'], m[col]['data']['dy'], colors[2*n+2]+':.', label=col) 
    plt.plot(m[col]['fit']['t'], m[col]['fit']['dy'], colors[2*n+3]+'-',label=col+' logistic fit')               
    plt.ylabel('Antal per dag')
    plt.legend(loc=1)

#plt.savefig('Deathplot_swe.png')
