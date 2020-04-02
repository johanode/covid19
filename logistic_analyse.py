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


#%% Read JH
# Data from John Hopkins University
# https://systems.jhu.edu/
# The Lancet: https://doi.org/10.1016/S1473-3099(20)30120-1
# Github https://github.com/CSSEGISandData/COVID-19

purl = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
csv_cases = pd.read_csv(purl+'time_series_covid19_confirmed_global.csv',sep=',',header=0,index_col=False)
csv_deaths = pd.read_csv(purl+'time_series_covid19_deaths_global.csv',sep=',',header=0,index_col=False)

# Create time serias data
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

#%% Plot selected countries
countries = ['Italy','Spain','US','France','Germany','Finland','Norway','Denmark','Sweden']

fig,axes = plt.subplots(nrows=2,ncols=2)
df_cases.loc[:,countries].plot(ax=axes[0,0],title='Confirmed cases')
df_deaths.loc[:,countries].plot(ax=axes[1,0],title='Deaths')
df_cases.loc[:,countries].plot(ax=axes[0,1],title='Confirmed cases')
df_deaths.loc[:,countries].plot(ax=axes[1,1],title='Deaths')

fig,axes = plt.subplots(nrows=2,ncols=2)
df_cases.loc[:,countries].diff().plot(ax=axes[0,0],title='Confirmed new cases')
df_deaths.loc[:,countries].diff().plot(ax=axes[1,0],title='New deaths')
df_cases.loc[:,countries].diff().plot(ax=axes[0,1],title='Confirmed new cases')
df_deaths.loc[:,countries].diff().plot(ax=axes[1,1],title='New deaths')


#%% SUBFUNCTIONS
def logistic_mdl(x,a,b,c):
    #a refers to the infection speed
    #b is the day with the maximum infections occurred
    #c is the total number of recorded infected people at the infection’s end
    return c/(1+np.exp(-(x-b)/a))
    
def prepare(df, country, xhat=np.arange(0,100)):
    dt=df.index-df.index[0]
    x0 = dt.days.values
    
    ys = df.loc[:,country]
    
    # set x=0 when first case is reported
    i = np.where(ys.values>0)[0]
    
    y = ys.values[i]
    dy = ys.diff().values[i]
    
    x = x0[i]-x0[i[0]]
    t = df.index.values[i]
    
    that = t
    while len(that)<len(xhat):
        that = np.append(that,that[-1]+np.timedelta64(1,'D'))
    
    return {'x':x, 'y':y, 't':t, 'xhat':xhat, 'that':that, 'dy':dy}    

#%% 
def fit(df, data_label=None, countries=['Sweden'], p0=None):
    # Init plot
    Nc = len(countries)
    nax = (2,Nc)    
    fig, axes = plt.subplots(nrows=nax[0], ncols=nax[1])
     
    output = {} 
    for n,country in enumerate(countries):
        # Data preperation for fit
        data = prepare(df,country)
        
        # Plot data
        plt.subplot(nax[0],nax[1],n+1)
        plt.plot(data['t'], data['y'], '-', label=data_label)
       
        try:
            # Logistic function fit (i.e. cdf of logistic distribution)
            popt, pcov = curve_fit(logistic_mdl, data['x'], data['y'], p0=p0)
            data['yhat'] = logistic_mdl(data['xhat'], *popt)
            
            plt.plot(data['that'], data['yhat'], 'r--', 
                     label='logistic fit (speed=%5.2f)'%popt[0])
        
        except:
            print('Error fitting '+country)
        
        plt.legend()  
        plt.title(country)
        
        output[country] = {
                'data':data,
                'mdl':{
                        'fun':logistic_mdl,
                        'p':popt
                        }
                }
    
    # Ploting per day and logistic pdf
    for n,country in enumerate(countries):
        data = output[country]['data']
        popt = output[country]['mdl']['p']
        
        # Plot cases per day
        plt.subplot(nax[0],nax[1],n+1+Nc)
        plt.bar(data['t'], data['dy'], label=data_label+' per day')
        
        # Plot logistic pdf
        dyhat = popt[2]*stats.logistic.pdf(data['xhat'],loc=popt[1],scale=popt[0])
        plt.plot(data['that'], dyhat, 'r--', label='fit (logistic pdf)')
        
        plt.legend()
    
    return output
         
    #%%
#countries = ['ChinaHubei','Italy','Spain','France','US','United Kingdom']
#countries = ['Italy','Germany','Finland','Norway','Denmark','Sweden']
countries=['Sweden','Italy']

#m = fit(df_cases, data_label='Confirmed cases', countries=countries, p0=[5,50,10000])
m = fit(df_deaths, data_label='Deaths', countries=countries, p0=[5,20,2000])
#plt.savefig('Deathplot.png')
