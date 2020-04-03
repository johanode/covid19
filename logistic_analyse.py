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

last_update = pd.to_datetime(csv_cases.columns[-1]).strftime('%Y-%m-%d')

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

#%%
#set x=0 when first case is reported
first_day_case = {}
for country in df_cases.columns:
    is_cases = df_cases[country].values>0
    if is_cases.any():
        first_day_case[country] = np.where(is_cases)[0][0]
    else:
        first_day_case[country] = []
    

#%% Plot selected countries
countries = ['Italy','Spain','US','France','Germany','Finland','Norway','Denmark','Sweden']

fig,axes = plt.subplots(nrows=2,ncols=2)
df_cases.loc[:,countries].plot(ax=axes[0,0],title='Confirmed cases')
df_deaths.loc[:,countries].plot(ax=axes[1,0],title='Deaths')
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
    i = np.arange(first_day_case[country],len(ys))    
    
    y = ys.values[i]
    dy = ys.diff().values[i]
    
    x = x0[i]-x0[i[0]]
    t = df.index.values[i]
    
    that = t
    while len(that)<len(xhat):
        that = np.append(that,that[-1]+np.timedelta64(1,'D'))
    
    return {
            'data':{'x':x, 'y':y, 't':t, 'dy':dy}, 
            'fit':{'x':xhat, 't':that}
            }

def fit(df, data_label=None, countries=['Sweden'], p0=None):
    output = {} 
    for n,country in enumerate(countries):
        # Data preperation for fit
        output[country] = prepare(df,country)
        try:
            # Logistic function fit (i.e. cdf of logistic distribution)
            popt, pcov = curve_fit(logistic_mdl, output[country]['data']['x'], output[country]['data']['y'], p0=p0)
            output[country]['fit']['y'] = logistic_mdl(output[country]['fit']['x'], *popt) 
            output[country]['mdl'] = {'fun':logistic_mdl, 'p':popt}
        except:
            output[country]['fit']['yhat'] = []
            output[country]['mdl'] = {'fun':logistic_mdl, 'p':[]}
            print('Error fitting '+country)
        
    return output

def plot(mfit, data_label=None, countries=None):
    
    if countries is None or len(countries)==0:
        countries = mfit.keys()
    
    # Init plot
    Nc = len(countries)
    nax = (2,Nc)    
    fig, axes = plt.subplots(nrows=nax[0], ncols=nax[1])
     
    for n,country in enumerate(countries):
        # Plot data
        plt.subplot(nax[0],nax[1],n+1)
        plt.plot(mfit[country]['data']['t'], mfit[country]['data']['y'], '-', label='data')
        plt.ylabel(data_label)
       
        popt = mfit[country]['mdl']['p']
        if len(popt)>0:
            plt.plot(mfit[country]['fit']['t'], mfit[country]['fit']['y'], 'r--', 
                    label='logistic fit (speed=%5.2f)'%popt[0])
        
        plt.legend()  
        plt.title(country)
        
        
        # Plot cases per day
        plt.subplot(nax[0],nax[1],n+1+Nc)
        plt.bar(mfit[country]['data']['t'], mfit[country]['data']['dy'], label='data')
        plt.ylabel(data_label+' per day')
        
        # Plot logistic pdf
        if len(popt)>0:
            dyhat = popt[2]*stats.logistic.pdf(mfit[country]['fit']['x'],loc=popt[1],scale=popt[0])
            plt.plot(mfit[country]['fit']['t'], dyhat, 'r--', label='fit (logistic pdf)')
        
        plt.legend()

    #%%
#countries = ['ChinaHubei','Italy','Spain','France','US','United Kingdom']
#countries = ['Italy','Germany','Finland','Norway','Denmark','Sweden']
countries=['Sweden','Italy']

#m = fit(df_cases, countries=countries, p0=[5,50,20000])
m = fit(df_deaths, countries=countries, p0=[5,30,2000])
plot(m,data_label='Deaths')

#plt.savefig('Deathplot.png')


