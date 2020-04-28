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
i_would_like_to_save_data_to_local_file = 'yes' #(yes/no)
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
                

# Confirmed cases
df_cases = xl.parse('Antal per dag region') #pd.read_excel(url,sheet_name=0)
df_cases.index = pd.to_datetime(df_cases['Statistikdatum'])
last_update = pd.to_datetime(df_cases['Statistikdatum'][-1])
df_cases.drop(['Statistikdatum'],axis=1,inplace=True)

# Replace missing values with zero
df_cases.fillna(0,inplace=True)  

# Deaths
df_deaths = xl.parse('Antal avlidna per dag') #pd.read_excel(url,sheet_name=1)
df_deaths.iloc[[isinstance(d,str) and 'saknas' in d for d in df_deaths['Datum_avliden'].values],0]=''
df_deaths.index = pd.to_datetime(df_deaths['Datum_avliden'])
df_deaths.drop(['Datum_avliden'],axis=1,inplace=True)

# Intesive care
df_iva = xl.parse('Antal intensivvårdade per dag') #pd.read_excel(url,sheet_name=2)
df_iva.index = pd.to_datetime(df_iva['Datum_vårdstart'])
df_iva.loc[:,'Antal_intensivvårdade'].fillna(0,inplace=True)  
df_iva.drop(['Datum_vårdstart'],axis=1,inplace=True)

xl.close()

#%% Merge to one dataframe
# Align index names
df_cases.index.name = 'Datum'
df_deaths.index.name = 'Datum'
df_iva.index.name = 'Datum'

# Merge
df = df_cases.copy()
df[df_deaths.columns[0]] = df_deaths.iloc[:,0]
df[df_iva.columns[0]] = df_iva.iloc[:,0]

# Fill missing to zero
df.fillna(0, inplace=True)  

# Renamce region column names
new_cols = {col : col+'_antal_fall' for col in df_cases if not 'fall' in col}
df.rename(columns=new_cols, inplace=True)

# set x=0 when first case is reported and exclude last date
for col in df:
    if 'antal_fall' in col:
        valid_cases = np.logical_and(df[col].cumsum().values>0,df.index<last_update)
        df[col.split('_')[0]+'_filter'] = valid_cases
        
#%% Plot selected columns
fig,axes = plt.subplots(nrows=2,ncols=2)
cols = ['Totalt_antal_fall', 'Stockholm_antal_fall']
df.loc[:,cols].cumsum().plot(ax=axes[0,0])

df.loc[:,cols].rolling(7).mean().plot(ax=axes[0,1])
plt.ylabel('Antal per dag (rull=7)')

cols = ['Antal_avlidna','Antal_intensivvårdade']
df.loc[:,cols].cumsum().plot(ax=axes[1,0])

df.loc[:,cols].rolling(7).mean().plot(ax=axes[1,1])
plt.ylabel('Antal per dag (rull=7)')

fig,axes = plt.subplots(nrows=1,ncols=2)
#cols = [x for x in df.columns if x not in cols]
cols = [new_cols[col] for col in ['Västra_Götaland', 'Uppsala', 'Skåne', 'Västerbotten', 'Norrbotten']]
df.loc[:,cols].cumsum().plot(ax=axes[0])
df.loc[:,cols].rolling(7).mean().plot(ax=axes[1])
plt.ylabel('Antal per dag (rull=7)')

#%% Plot selected columns with respect to poulation
population ={
        'Norrbotten' : 250093,
        'Skåne' : 1377827,
        'Västra_Götaland' : 1725881,
        'Stockholm' : 2377081
        }

fig = plt.figure()
plt.subplot(211)
for col in list(population.keys()):
    s = df.loc[:,new_cols[col]].cumsum()/population[col]*100e3
    plt.plot(s.values,label=col)
plt.legend()
plt.ylabel('Case per 100k')
    
plt.subplot(212)
for col in list(population.keys()):    
    s = df.loc[:,new_cols[col]].rolling(7).mean()/population[col]*100e3
    plt.plot(s.values,label=col) 
plt.ylabel('Case per day 100k')
plt.legend()   

#%% SUBFUNCTIONS
def logistic_mdl(x,a,b,c):
    #a refers to the infection speed
    #b is the day with the maximum infections occurred
    #c is the total number of recorded infected people at the infection’s end
    return c/(1+np.exp(-(x-b)/a))

def logistic_b_mdl(x,a,c):
    global b
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
    if 'antal_fall' in col:
        i = np.where(df[col.split('_')[0]+'_filter'])[0]
    else:
        i = np.where(df['Totalt_filter'])[0]        
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

def fit(df, mdl=logistic_mdl, data_label=None, cols=[0], p0=None):    
    output = {}         
    if isinstance(cols,str):
        cols = [cols]
    for n,col in enumerate(cols):
        # Data preperation for fit
        output[col] = prepare(df,col)
        
        try:
            # Logistic function fit (i.e. cdf of logistic distribution)
            popt, pcov = curve_fit(mdl, output[col]['data']['x'], output[col]['data']['y'], p0=p0)
            output[col]['fit']['y'] = mdl(output[col]['fit']['x'], *popt) 
            output[col]['mdl'] = {'fun':mdl, 'p':popt}                                    
        except:
            output[col]['fit']['y'] = []
            output[col]['mdl'] = {'fun':mdl, 'p':[]}            
            print('Error fitting'+col)     
        
        try:
            # Per day and logistic pdf          
            if mdl==logistic_mdl:   
                output[col]['fit']['dy'] = popt[2]*stats.logistic.pdf(output[col]['fit']['x'],loc=popt[1],scale=popt[0])                   
            elif mdl==logistic_b_mdl:
                global b
                output[col]['fit']['dy'] = popt[1]*stats.logistic.pdf(output[col]['fit']['x'],loc=b,scale=popt[0])
            else:
                output[col]['fit']['dy'] = []        
        except:
            output[col]['fit']['dy'] = []
                
                
    return output
  
#%% Fit logistic model and save to dataframe
m_1 = fit(df, data_label='Antal fall', cols=['Totalt_antal_fall'], p0=[5,50,10000]) 
m_2 = fit(df, data_label='Antal', cols=['Antal_avlidna','Antal_intensivvårdade'], p0=[5, 70, 2000]) #[4,50,1000]
m = {**m_1,**m_2} 

#%% Fit logistic model for different days
cols = ['Totalt_antal_fall','Antal_avlidna','Antal_intensivvårdade']
for r in ['Stockholm','Västra_Götaland', 'Uppsala', 'Skåne', 'Västerbotten', 'Norrbotten']:
    cols += [new_cols[r]]

yhat = pd.DataFrame()
t = []
for dt in range(0,50):
    d = pd.to_datetime('2020-04-01')+pd.to_timedelta(dt*24*60*60*1e9)
    if d>last_update:
        break
    t.append(d)
    print(d.strftime('%Y-%m-%d'))    
    m_d = fit(df.loc['2020-02-01':d.strftime('%Y-%m-%d')], data_label='Antal', cols=cols, p0=[5, 70, 2000]) #[4,50,1000]
    for col in cols:
        if len(m_d[col]['mdl']['p'])>0:
            yhat.loc[d,col] = m_d[col]['mdl']['p'][2]
        else:
            yhat.loc[d,col] = np.nan

#%%
fig,axes = plt.subplots(nrows=2,ncols=1)
cols2 = ['Antal_avlidna','Antal_intensivvårdade']
cols1 = np.setdiff1d(cols,cols2)
yhat[cols1].plot(ax=axes[0],logy=True)
yhat[cols2].plot(ax=axes[1])

#%% Fit logistic model with fix offset
cols = ['Antal_intensivvårdade','Antal_avlidna']
db = [5,5+2]

b = np.round(m_1['Totalt_antal_fall']['mdl']['p'][1]+db[0])
m_b_1 = fit(df, data_label='Antal', cols=cols[0], mdl=logistic_b_mdl, p0=[5, 2500]) 

b = np.round(m_1['Totalt_antal_fall']['mdl']['p'][1]+db[1])
m_b_2 = fit(df, data_label='Antal', cols=cols[1], mdl=logistic_b_mdl, p0=[5, 2500]) 

m_b = {**m_b_1,**m_b_2}


#%% plot
cols = ['Totalt_antal_fall']
# Init plot
fig,axes = plt.subplots(nrows=2,ncols=1)
colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8']
# Plot data  
ax11 = plt.subplot(211)  
for col in cols: 
    plt.plot(m[col]['data']['t'], m[col]['data']['y'], colors[0]+':.', label=col)         
    plt.plot(m[col]['fit']['t'], m[col]['fit']['y'], colors[1]+'-',label=col+' logistic fit')    
plt.ylabel('Antal fall')
plt.legend(loc=2)

ax12 = ax11.twinx()
for n,col in enumerate(['Antal_intensivvårdade','Antal_avlidna']): 
    plt.plot(m[col]['data']['t'], m[col]['data']['y'], colors[3*n+3]+':.', label=col) 
    plt.plot(m[col]['fit']['t'], m[col]['fit']['y'], colors[3*n+4]+'-',label=col+' logistic fit')  
    #plt.plot(m_b[col]['fit']['t'], m_b[col]['fit']['y'], colors[3*n+4]+'--',label=col+(' logistic fit (b=+%.1f)' % db[n]))
plt.ylabel('Antal iva/avlidna')
plt.legend(loc=6)

    
ax21 = plt.subplot(212)  
for col in cols: 
    plt.plot(m[col]['data']['t'], m[col]['data']['dy'], colors[0]+':.', label=col)         
    plt.plot(m[col]['data']['t'], pd.Series(m[col]['data']['dy']).rolling(7).mean().values, colors[1]+'-', label=col) 
    plt.plot(m[col]['fit']['t'], m[col]['fit']['dy'], colors[2]+'--',label=col+' logistic fit')          
plt.ylabel('Antal fall per dag')
plt.legend(loc=2)

ax22 = ax21.twinx()
for n,col in enumerate(['Antal_intensivvårdade','Antal_avlidna']): 
    plt.plot(m[col]['data']['t'], m[col]['data']['dy'], colors[3*n+3]+'.:', label=col) 
    plt.plot(m[col]['data']['t'], pd.Series(m[col]['data']['dy']).rolling(7).mean().values, colors[3*n+4]+'-', label=col) 
    plt.plot(m[col]['fit']['t'], m[col]['fit']['dy'], colors[3*n+5]+'--',label=col+' logistic fit')               
    #plt.plot(m_b[col]['fit']['t'], m_b[col]['fit']['dy'], colors[4*n+5]+'--',label=col+(' logistic fit (b=+%.1f)' % db[n]))              
plt.ylabel('Antal iva/avlidna per dag')
plt.legend(loc=6)

#%% Plot logistic model with fix offset
#fig,axes = plt.subplots(nrows=1,ncols=1)
#c = {'db':list(range(-7,15))}
#cols = ['Antal_intensivvårdade','Antal_avlidna']
#for n,col in enumerate(cols):        
#    c[col]=[]
#    for o in c['db']: #offset from cases        
#        b = np.round(m_1['Totalt_antal_fall']['mdl']['p'][1]+o)
#        m_o = fit(df, data_label='Antal', cols=cols, mdl=logistic_b_mdl, p0=[5, 2500]) 
#        c[col].append(m_o[col]['mdl']['p'][1])
#    plt.plot(c['db'],c[col],colors[n],label=col)
#    plt.plot(c['db'],[m_2[col]['mdl']['p'][2]]*len(c['db']),colors[n]+'--',label=col)
#plt.legend()
#plt.xlabel('Offset b')
#plt.ylabel('Antal')
