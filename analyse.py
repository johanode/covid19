# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:59:42 2020

@author: Johan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%% Read JH
p = 'C:/Users/Johan/Documents/Python Scripts/COVID-19/csse_covid_19_data/csse_covid_19_time_series/'
csv_cases = pd.read_csv(p+'time_series_covid19_confirmed_global.csv',sep=',',header=0,index_col=False)
csv_deaths = pd.read_csv(p+'time_series_covid19_deaths_global.csv',sep=',',header=0,index_col=False)

def totimeseries(df):
    data = df.iloc[:,5:].transpose()
    country = df['Country/Region'].values
    state = df['Province/State'].values
    state[df['Province/State'].isna().values]=''
    

    colnames = [a+b for a,b in zip(country,state)]
    data.columns=colnames    
    data.index = pd.to_datetime(data.index)
    return data

df_deaths = totimeseries(csv_deaths)
df_cases = totimeseries(csv_cases)

#%%
cols = ['Italy','Spain','US','France','Germany','Finland','Norway','Denmark','Sweden']
global_countries = ['ChinaHubei','Italy','Spain','France','Germany','US','United Kingdom']
local_countries = ['Finland','Norway','Denmark','Sweden']
all_countries = ['ChinaHubei','Italy','Spain','France','US','United Kingdom', 
                 'Germany','Netherlands','Finland','Norway','Denmark','Sweden']

fig,axes = plt.subplots(nrows=2,ncols=2)
df_cases.loc[:,global_countries].plot(ax=axes[0,0],title='Confirmed cases')
df_deaths.loc[:,global_countries].plot(ax=axes[1,0],title='Deaths')
df_cases.loc[:,local_countries].plot(ax=axes[0,1],title='Confirmed cases')
df_deaths.loc[:,local_countries].plot(ax=axes[1,1],title='Deaths')

fig,axes = plt.subplots(nrows=2,ncols=2)
df_cases.loc[:,global_countries].diff().plot(ax=axes[0,0],title='Confirmed new cases')
df_deaths.loc[:,global_countries].diff().plot(ax=axes[1,0],title='New deaths')
df_cases.loc[:,local_countries].diff().plot(ax=axes[0,1],title='Confirmed new cases')
df_deaths.loc[:,local_countries].diff().plot(ax=axes[1,1],title='New deaths')


#%% Read sweden
url = 'https://www.arcgis.com/sharing/rest/content/items/b5e7488e117749c19881cce45db13f7e/data'
df_swe = pd.read_excel(url,sheet_name=1)


#%%
def exp_mdl(x, a, b, c, d):
    return a * np.exp(b * (x-c)) + d

def linear_mdl(x, m, b):
    return m*x + b

def logistic_mdl(x,a,b,c):
    #a refers to the infection speed
    #b is the day with the maximum infections occurred
    #c is the total number of recorded infected people at the infection’s end
    return c/(1+np.exp(-(x-b)/a))
    

def prepare(df,country,xhat=np.arange(0,100)):
    dt=df.index-df.index[0]
    x0 = dt.days.values
    
    y = df.loc[:,country].values
    i = np.where(y>0)[0]
    x = x0[i]-x0[i[0]]
    y = y[i]
    
    t = df.index.values[i]
    
    that = t
    while len(that)<len(xhat):
        that = np.append(that,that[-1]+np.timedelta64(1,'D'))
    
    return {'x':x, 'y':y, 't':t, 'xhat':xhat, 'that':that}    

def fit(data, mdl=logistic_mdl, p0=None, return_mdl=False):
    popt, pcov = curve_fit(mdl, data['x'], data['y'], p0=p0)
    if return_mdl:
        return popt, mdl
    else:
        return mdl(data['xhat'], *popt), popt

def logistic_b_mdl(x,a,c):
    global b
    return c/(1+np.exp(-(x-b)/a))

def logistic_a_mdl(x,b,c):
    global a
    return c/(1+np.exp(-(x-b)/a))


#%%
global_countries = ['ChinaHubei','Italy','Spain','France','US','United Kingdom']
local_countries = ['Italy','Germany','Finland','Norway','Denmark','Sweden']

df = df_cases.copy()
data_label = 'Confirmed cases'

pred = {}
for country in ['ChinaHubei','Italy']:
    data = prepare(df,country)
    pred[country],mdl = fit(data,return_mdl=True)
#p0 = pred['Italy']
p0 = [5,50,10000]


fig, axes = plt.subplots(nrows=3, ncols=2)
for n,country in enumerate(local_countries):
    print(country)
    data = prepare(df,country)
    
    plt.subplot(axes.shape[0],axes.shape[1],n+1)
    
    plt.plot(data['t'], data['y'], '-', label=data_label)
   
    
    try:
        data['yhat'],popt = fit(data, p0=p0)
        print(popt)
        plt.plot(data['that'], data['yhat'], 'r--', label='logistic fit (speed=%5.2f)'%popt[0])
    except:
        None
    
    for refcountry in ['Italy']:
        a = pred[refcountry][0]
        b = pred[refcountry][1]
        if np.logical_and(country != refcountry, country != 'ChinaHubei'): 
            try:
                yhat2,popt2 = fit(data, mdl=logistic_a_mdl, p0=popt[1:])
                plt.plot(data['that'], yhat2, 'y:', label='logistic fit (speed=%s)' % refcountry)
                #print(popt2)
            except:
                None
                
    plt.legend()  
    plt.title(country)
    
    
#%% Deaths
df = df_deaths.copy()
data_label = 'Deaths' #'Confirmed cases'

pred = {}
for country in ['ChinaHubei','Italy']:
    data = prepare(df,country)
    pred[country],mdl = fit(data,return_mdl=True)
#p0 = pred['Italy']
p0 = [4,20,2000] 

fig, axes = plt.subplots(nrows=3, ncols=2)
for n,country in enumerate(local_countries):
    print(country)
    data = prepare(df,country)
    
    plt.subplot(axes.shape[0],axes.shape[1],n+1)
    
    plt.plot(data['t'], data['y'], '-', label=data_label)
   
    
    try:
        data['yhat'],popt = fit(data, p0=p0)
        print(popt)
        plt.plot(data['that'], data['yhat'], 'r--', label='logistic fit (speed=%5.2f)'%popt[0])
    except:
        None
    
    for refcountry in ['Italy']:
        a = pred[refcountry][0]
        b = pred[refcountry][1]
        if np.logical_and(country != refcountry, country != 'ChinaHubei'): 
            try:
                yhat2,popt2 = fit(data, mdl=logistic_b_mdl, p0=popt[0::2])
                plt.plot(data['that'], yhat2, 'y:', label='logistic fit (max=%s)' % refcountry)
                #print(popt2)
            except:
                None
                
    plt.legend()  
    plt.title(country)