# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:59:42 2020

@author: Johan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

#%% Read JH daily report
p = 'C:/Users/Johan/Documents/Python Scripts/COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/'
csv_d = pd.read_csv(p+'04-01-2020.csv',sep=',',header=0,index_col=False)


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
df_swe = pd.read_excel(url,sheet_name=0)
df_swe.index = pd.to_datetime(df_swe['Statistikdatum'])
df_swe_d = pd.read_excel(url,sheet_name=1)

filename = 'Antal nyinskrivna vårdtillfällen med Coronavirus - Period 2020-01-01 - 2020-04-02.xlsx'
df_swe_iva = pd.read_excel(filename,header=1)


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
    
def prepare(df, country, diff=False, xhat=np.arange(0,100)):
    dt=df.index-df.index[0]
    x0 = dt.days.values
    
    y = df.loc[:,country]
    i = np.where(y.values>0)[0]
    if diff:
        y = y.diff().values[i]
    else:
        y = y.values[i]
    x = x0[i]-x0[i[0]]
    t = df.index.values[i]
    
    that = t
    while len(that)<len(xhat):
        that = np.append(that,that[-1]+np.timedelta64(1,'D'))
    
    return {'x':x, 'y':y, 't':t, 'xhat':xhat, 'that':that}    

def fit(data, mdl=logistic_mdl, p0=None, return_mdl=False):
    if 'poly' in mdl:
        global npoly
        popt = np.polyfit(data['x'], data['y'], npoly)
        
        if return_mdl:
            return popt, npoly
        else:
            return np.polyval(popt,data['xhat']), popt
    else:
        if isinstance(mdl,str):
            if 'logistic' in mdl:
                mdl=logistic_mdl
            elif 'exp' in mdl:
                mdl = exp_mdl
            else:
                mdl = linear_mdl
                
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
def run(df, data_label=None, countries=['Sweden'], refcountries=[], p0=None, diff=False, model='logsitic', plot='plot'):
    
    if isinstance(p0,str):
        p0ref = [p0]
    else:
        p0ref = []
        
    pred = {}
    for country in refcountries+p0ref:
        data = prepare(df,country,diff=False)
        pred[country],mdl = fit(data, mdl=model, return_mdl=True)
    
    if isinstance(p0,str):
        p0 = pred[p0ref]
    
    Nc = len(countries)
#    if Nc<=3:
#        nax = (Nc,1)
#    elif Nc<=6:
#        nax = (np.ceil(Nc/2), 2)
#    else:
#        nax = (np.ceil(Nc/3), 3)
    nax = (Nc,2)    
    if plot is not None:
        fig, axes = plt.subplots(nrows=nax[0], ncols=nax[1])
     
    output = {} 
    for n,country in enumerate(countries):
        #print(country)
        data = prepare(df,country,diff=False)
       
        if plot is not None:
            plt.subplot(nax[0],nax[1],n+1)
            if 'bar' in plot:
                plt.bar(data['t'], data['y'], label=data_label)
            else:
                plt.plot(data['t'], data['y'], '-', label=data_label)
       
        
        try:
            data['yhat'],popt = fit(data, mdl=model, p0=p0)
            #print(popt)
            if 'logistic' in model:
                label = model+' fit (speed=%5.2f)'%popt[0]
            else:
                label = model+' fit'
            i = np.logical_and(data['yhat']>=0,data['yhat']<1e6)
            #if 'bar' in plot:
            #    plt.bar(data['that'][i], data['yhat'][i], label=label)
            if plot is not None:
                plt.plot(data['that'][i], data['yhat'][i], '--', label=label)
                
        except:
            None
        
        for refcountry in refcountries:
            if np.logical_and(country != refcountry, country != 'ChinaHubei'): 
                try:
                    if 'logistic' in model:
                        a = pred[refcountry][0]
                        b = pred[refcountry][1]
                        yhat2,popt2 = fit(data, mdl=logistic_a_mdl, p0=popt[1:])
                        label=model+' fit (speed=%s)' % refcountry
                    elif 'poly' in model:
                        print(refcountry)
                        yhat2 = np.polyval(pred[refcountry],data['xhat'])
                    else:
                        yhat2 = []
                        label=model+' fit (%s)' % refcountry
                    #if 'bar' in plot:
                    #    plt.bar(data['that'], yhat2, label=label)
                    if plot is not None:
                        plt.plot(data['that'], yhat2, 'g:', label=label)
                    #print(popt2)
                except:
                    None
                    
        plt.legend()  
        plt.title(country)
        
        output[country] = {'data':data,'mdl':{'name':model,'p':popt}}
    
    for n,country in enumerate(countries):
        #print(country)
        plt.subplot(nax[0],nax[1],n+1+Nc)
        data = prepare(df,country,diff=True)
        popt = m[country]['mdl']['p']
        plt.plot(data['t'], data['y'], '-', label=data_label+' per day')
        ytilde = popt[2]*stats.logistic.pdf(data['xhat'],loc=popt[1],scale=popt[0])
        plt.plot(data['that'][i], ytilde, '--', label='fit')
        plt.legend()
    return output
         
    #%%
global_countries = ['ChinaHubei','Italy','Spain','France','US','United Kingdom']
local_countries = ['Italy','Germany','Finland','Norway','Denmark','Sweden']

#p0=[4,20,2000]
#p0=[5,50,10000]
npoly=4
m = run(df_deaths, data_label='Deaths', diff=False, model='logistic', p0=[5,20,2000],
        plot='yes',
        countries=['Italy','Sweden'])
