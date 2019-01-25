# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 12:08:40 2018

@author: rbgud
"""

import matplotlib.cm as cm
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from scipy import stats
import networkx as nx
import sys
sns.set()

#%%

HFund = pd.read_csv('C:/Users/rbgud/OneDrive/바탕 화면/가을학기 후반/빅데이터와 금융자료분석/hedge (1).csv',engine='python')
HFund.index = pd.to_datetime(HFund['DATE'],format='%Y-%m-%d')
HFund.pop('DATE')

Bank = pd.read_csv('C:/Users/rbgud/OneDrive/바탕 화면/가을학기 후반/빅데이터와 금융자료분석/bank (1).csv',engine='python')
Bank.index = pd.to_datetime(Bank['DATE'],format='%Y-%m-%d')
Bank.pop('DATE')

Insurance = pd.read_csv('C:/Users/rbgud/OneDrive/바탕 화면/가을학기 후반/빅데이터와 금융자료분석/insurance (1).csv',engine='python')
Insurance.index = pd.to_datetime(Insurance['DATE'],format='%Y-%m-%d')
Insurance.pop('DATE')

Bdealer = pd.read_csv('C:/Users/rbgud/OneDrive/바탕 화면/가을학기 후반/빅데이터와 금융자료분석/dealer (1).csv',engine='python')
Bdealer.index = pd.to_datetime(Bdealer['Date'],format='%Y-%m-%d')
Bdealer.pop('Date')

mkc=pd.read_csv('C:/Users/rbgud/OneDrive/바탕 화면/가을학기 후반/빅데이터와 금융자료분석/mkc.csv', index_col='DATE',engine='python')
mkc.index=pd.to_datetime(mkc.index,format='%Y-%m-%d')


#%% Set data period 

start_date = '2000-05-31'

HFund = HFund.loc[HFund.index >= start_date]

Bank = Bank.loc[Bank.index >= start_date]

Insurance = Insurance.loc[Insurance.index >= start_date]

Bdealer = Bdealer.loc[Bdealer.index >= start_date]

#%% Make Return data 

HFund_ret = HFund.pct_change().dropna(axis=0,how='all').dropna(axis=1,how='any').iloc[:,:]

Bank_ret = Bank.pct_change().dropna(axis=0,how='all').dropna(axis=1,how='any').iloc[:,:25]

Insurance_ret = Insurance.pct_change().dropna(axis=0,how='all').dropna(axis=1,how='any').iloc[:,:25]

Bdealer_ret = Bdealer.pct_change().dropna(axis=0,how='all').dropna(axis=1,how='any').iloc[:,:]

Bdealer_ret.pop('ISTN US Equity')
Bdealer_ret.pop('NDB US Equity')

#%% Concate total data 

Data_total = pd.concat([Bank_ret,Bdealer_ret,HFund_ret,Insurance_ret],axis=1).dropna(axis=1,how='any')

#%%

def make_name(data):
    name=dict(zip(range(len(data)),data.columns))
    name_buff=dict(zip(data.columns,range(len(data))))
    return name, name_buff

bank_name, bank_name_buff = make_name(Bank_ret)

dealer_name, dealer_name_buff=make_name(Bdealer_ret)

insurance_name, insurance_name_buff= make_name(Insurance_ret)

hedge_name, hedge_name_buff= make_name(HFund_ret)

name, name_buff= make_name(Data_total)

#%% Make subperiod

subperiod1_data = Data_total.loc[Data_total.index <='2003-12-31']
subperiod2_data = Data_total.loc[(('2004-01-01'<= Data_total.index)&(Data_total.index<='2006-12-31'))]
subperiod3_data = Data_total.loc[(('2007-01-01'<= Data_total.index)&(Data_total.index<='2009-12-31'))]
subperiod4_data = Data_total.loc[(('2010-01-01'<= Data_total.index)&(Data_total.index<='2012-12-31'))]
subperiod5_data = Data_total.loc[(('2013-01-01'<= Data_total.index)&(Data_total.index<='2015-12-31'))]
subperiod6_data = Data_total.loc[(('2016-01-01'<= Data_total.index)&(Data_total.index<='2018-12-31'))]

#%% Make subperiod 2

subperiod1_data_ = Data_total.loc[Data_total.index <='2001-12-31']
subperiod2_data_ = Data_total.loc[(('2002-01-01'<= Data_total.index)&(Data_total.index<='2004-12-31'))]
subperiod3_data_ = Data_total.loc[(('2006-01-01'<= Data_total.index)&(Data_total.index<='2008-12-31'))]
subperiod4_data_ = Data_total.loc[(('2009-01-01'<= Data_total.index)&(Data_total.index<='2011-12-31'))]
subperiod5_data_ = Data_total.loc[(('2012-01-01'<= Data_total.index)&(Data_total.index<='2015-12-31'))]
subperiod6_data_ = Data_total.loc[(('2018-01-01'<= Data_total.index)&(Data_total.index<='2018-12-31'))]

#%%
def risk_of_system(data, i=-1):
    pca=PCA()
    pca.fit(data)
    if i==-1:
        return np.array(sorted(pca.explained_variance_, reverse=True))[:].sum()
    else:
        return np.array(sorted(pca.explained_variance_, reverse=True))[:i].sum()
#%%
'''PCAS 함수 정의 및 비교, risk fraction 비교'''

def PCAS( data2, i, n, threshold):
    name_buff=dict(zip(data2.columns,range(len(data2.columns))))
    pca=PCA()
    pca.fit(data2)
    risk_fraction=risk_of_system(data2,n)/risk_of_system(data2)
    if risk_fraction >= threshold:
        empty=0
        for j in range(n):
            empty+=(np.array(sorted(pca.explained_variance_, reverse=True))[j]**2) * (np.array(pca.components_)[j][name_buff[i]]**2)
        return empty*( (data2.cov()[i][i]) / (data2.cov().sum().sum()) ) * (10**5)
    else:
        return print("risk fraction이 threshold를 넘지않음")

def make_table(data, start, end):
    table=pd.DataFrame(columns=["PCAS1", "PCAS1-10", "PCAS1-20"], index=['Mean', 'Min', 'Max'])
    for n in [1, 10, 20]:
        empty=[]
        if n > len(data.columns):
            n=len(data.columns)
            for i in data.columns:
                empty.append(PCAS( Data_total[start:end] ,i , n, 0.1))
            table.iloc[0, n//10+1]=float(pd.DataFrame(empty).mean().values)
            table.iloc[1, n//10+1]=float(pd.DataFrame(empty).min().values)
            table.iloc[2, n//10+1]=float(pd.DataFrame(empty).max().values)
        for i in data.columns:
            empty.append(PCAS( Data_total[start:end] ,i , n, 0.1))
        table.iloc[0, n//10]=float(pd.DataFrame(empty).mean().values)
        table.iloc[1, n//10]=float(pd.DataFrame(empty).min().values)
        table.iloc[2, n//10]=float(pd.DataFrame(empty).max().values)
    return table


def make_year_tabel(start, end):
    bank_table=make_table(Bank_ret, start,end)
    hedge_table=make_table(HFund_ret, start, end)
    dealer_table=make_table(Bdealer_ret, start, end)
    insurance_table=make_table(Insurance_ret, start, end)
    return pd.concat([bank_table, hedge_table, dealer_table, insurance_table])
    

table_2001_2002= make_year_tabel('2001', '2002')
table_2002_2004= make_year_tabel('2002', '2004')
table_2006_2008= make_year_tabel('2006', '2008')
table_2009_2011= make_year_tabel('2009', '2011')
table_2012_2014= make_year_tabel('2012', '2014')
table_2015_2017= make_year_tabel('2015', '2017')

table=pd.DataFrame(columns=["PC1", "PC1-10", "PC1-20"], 
                   index=['2000 to 2003', '2004 to 2006', '2007 to 2009' , 
                          '2010 to 2012', '2013 to 2015', '2016 to 2018'])
table.iloc[0,0]=risk_of_system(subperiod1_data,1)/risk_of_system(subperiod1_data)
table.iloc[0,1]=risk_of_system(subperiod1_data,10)/risk_of_system(subperiod1_data)
table.iloc[0,2]=risk_of_system(subperiod1_data,20)/risk_of_system(subperiod1_data)

table.iloc[1,0]=risk_of_system(subperiod2_data,1)/risk_of_system(subperiod2_data)
table.iloc[1,1]=risk_of_system(subperiod2_data,10)/risk_of_system(subperiod2_data)
table.iloc[1,2]=risk_of_system(subperiod2_data,20)/risk_of_system(subperiod2_data)

table.iloc[2,0]=risk_of_system(subperiod3_data,1)/risk_of_system(subperiod3_data)
table.iloc[2,1]=risk_of_system(subperiod3_data,10)/risk_of_system(subperiod3_data)
table.iloc[2,2]=risk_of_system(subperiod3_data,20)/risk_of_system(subperiod3_data)

table.iloc[3,0]=risk_of_system(subperiod4_data,1)/risk_of_system(subperiod4_data)
table.iloc[3,1]=risk_of_system(subperiod4_data,10)/risk_of_system(subperiod4_data)
table.iloc[3,2]=risk_of_system(subperiod4_data,20)/risk_of_system(subperiod4_data)

table.iloc[4,0]=risk_of_system(subperiod5_data,1)/risk_of_system(subperiod5_data)
table.iloc[4,1]=risk_of_system(subperiod5_data,10)/risk_of_system(subperiod5_data)
table.iloc[4,2]=risk_of_system(subperiod5_data,20)/risk_of_system(subperiod5_data)


table.iloc[5,0]=risk_of_system(subperiod6_data,1)/risk_of_system(subperiod6_data)
table.iloc[5,1]=risk_of_system(subperiod6_data,10)/risk_of_system(subperiod6_data)
table.iloc[5,2]=risk_of_system(subperiod6_data,20)/risk_of_system(subperiod6_data)
print(" Table 작성 완료 ")

#%%
window = 36
def draw_scree(data,i,window):
    draw=[]
    for j in range(len(data)-window):
        pca=PCA()
        pca.fit(data[j:j+window])
        draw.append(pca.explained_variance_ratio_[0:i].sum())
        
    return np.array(draw)   
    
plt.figure(figsize = (12,7))    
a=draw_scree(Data_total, 1,window)*100
b=draw_scree(Data_total, 5,window)*100
c=draw_scree(Data_total, 10,window)*100
d=draw_scree(Data_total, 25,window)*100
# Your x and y axis
#x=Data_total.index[30:216]
x=Data_total.index[36:]
y=[ a, b-a, c-b, d-c]
 
# use a known color palette (see..)
pal = sns.color_palette("Set1")
plt.stackplot(x,y, labels=['PC1','PC5','PC10','PC25'], colors=pal, alpha=0.4 )
plt.ylabel('Explained variance')
plt.xlabel('Date')
plt.legend(loc='upper left')
plt.title('Principal Component Analysis Explained Variance Ratio')
plt.legend(loc='upper left')
plt.show()
#%%
window=36
#def rolling_month_ret(data, window):
#    return roll_month_ret=month_ret.mean(axis=1)*100

#roll_month_ret=rolling_month_ret(Data_total, window)
roll_month_ret=Data_total.mean(axis=1)
def roll_garch(data, window):
    empty=[]
    for i in range(len(data)-window):
        gam11 = arch_model(data[i:i+window], p=1, q=1) 
        resg11 = gam11.fit()
        yhat = resg11.forecast(horizon=1)
        empty.append(yhat.variance.values[-1, :])
    return empty

vol_forecast=roll_garch(roll_month_ret, window)
vol_forecast=pd.DataFrame(vol_forecast, index=roll_month_ret[window:].index)

#%%

am = arch_model(roll_month_ret*100, vol='Garch', p=1, o=0, q=1, dist='Normal')
index = roll_month_ret.index
start_loc = 0
window = 40
forecasts = pd.DataFrame()
for i in range(len(roll_month_ret)-window):
    sys.stdout.write('.')
    sys.stdout.flush()
    res = am.fit(first_obs=i, last_obs=i+window, disp='off')
    temp = res.forecast(horizon=1).variance
    fcast = temp.iloc[i+window-1]#.values
#    forecasts[fcast.name] = fcast
    forecasts_ = fcast
    forecasts = pd.concat([forecasts,forecasts_],axis=1)

garch_forecast = forecasts.T/10

#%%
plt.figure(figsize = (12,7))
#plt.plot(vol_forecast.ix[:'2009-04-30'].index,vol_forecast.ix[:'2009-04-30'],label='GARCH(1,1) Volatility')
plt.plot(garch_forecast.index,garch_forecast,label='GARCH(1,1) Volatility')
plt.ylabel('Volatility')
plt.xlabel('Date')
plt.legend(loc='upper left')
plt.title('Volatility forecasting using GARCH(1,1)')

#%%
empty=[]
for i in range(len(roll_month_ret)-36):
    empty.append((roll_month_ret.iloc[i:i+36].std())**2)
empty=pd.DataFrame(empty, index=roll_month_ret[36:].index)

plt.figure(figsize = (12,7))
plt.plot(empty.index[:216], empty*10,label='Real Volatility Scaled') 
plt.plot(empty.index[:216], empty,label='Real Volatility Non-scaled')  
plt.plot(vol_forecast.index, vol_forecast,label='GARCH(1,1) Volatility')
plt.ylabel('Volatility')
plt.xlabel('Date')
plt.legend(loc='upper left')
plt.title('Volatility forecasting using GARCH(1,1)')

#%%

def get_PCAS(start, end, n):
    empty={}
    for i in Data_total.columns:
        empty[i]=PCAS( Data_total[start: end], i, n, 0.1)
    empty=pd.DataFrame([empty], index=['PCAS']).T
    return empty

PCAS1=get_PCAS('2002-10', '2005-09', 1).rank(ascending=False)
PCAS10=get_PCAS('2002-10', '2005-09', 10).rank(ascending=False)
PCAS20=get_PCAS('2002-10', '2005-09', 20).rank(ascending=False)

PCAS_1=get_PCAS('2004-07', '2007-06', 1).rank(ascending=False)
PCAS_10=get_PCAS('2004-07', '2007-06', 10).rank(ascending=False)
PCAS_20=get_PCAS('2004-07', '2007-06', 20).rank(ascending=False)

percent_loss=((mkc['2007-06'] - mkc['2007-07':'2008-12'].min()) / mkc['2007-06']).T

percent_loss.columns=['Max%LOSS']
percent_loss=percent_loss.rank(ascending=False)

dic={}
def reg_m(y, x):
    empty=pd.concat([y,x], axis=1)
    ones = np.ones(len(x))
    X = sm.add_constant(np.column_stack((empty['PCAS'], ones)))
    Y=np.array(empty['Max%LOSS'].values).reshape(len(y),1)
    results=sm.OLS(Y, X).fit()
    results.summary()
    tau, _ = stats.kendalltau(x,y) 
    return results.params[0], results.tvalues[0], results.pvalues[0], tau


dic['PCAS1']=reg_m(percent_loss, PCAS1)
dic['PCAS10']=reg_m(percent_loss, PCAS10)
dic['PCAS20']=reg_m(percent_loss, PCAS20)
dic['PCAS_1']=reg_m(percent_loss, PCAS_1)
dic['PCAS_10']=reg_m(percent_loss, PCAS_10)
dic['PCAS_20']=reg_m(percent_loss, PCAS_20)


reg_result=pd.DataFrame.from_dict(dic,orient='index', columns=['Coeff', 't-stat', 'p-value', 'Kendall t'])


#%%

def Grangercausality_test(data,reverse):
    
    Check_list = []
    Save_index = []
    Score_list = []
    Pvalue_list = []
    in_stock_list = []
    out_stock_list = []
    
    for i in range(data.shape[1]):
        
        if reverse == False : # Granger cause
            
           j_range = np.arange(i+1,data.shape[1])
           
        elif reverse == None:
            
           j_range = np.arange(data.shape[1]) 
           
        else: # Reverse Granger cause  
            
           j_range = np.arange(i+1) 
          
        for j in j_range:    
           
           maxlag=3
               
           if i!=j:
            
                result_ = grangercausalitytests(data[[data.columns[i],data.columns[j]]],
                                                maxlag=maxlag,
                                                addconst=True,
                                                verbose=False)
            
                '''Null reject case : 유의 수준을 0.05(5%)로 설정하였고 테스트를 통해서 검정값(p-value)가 0.05이하로 
                나오면 귀무가설을 기각할 수 있다. 귀무가설은 “Granger Causality를 따르지 않는다” 이다.'''
               
                if result_[3][0]['ssr_ftest'][1] < 0.05 : 
                    
#                   print('{} granger causes {}'.format(data.columns[j],data.columns[i]))
                    
                   Check_list.append(['{} granger causes {}'.format(data.columns[j],data.columns[i])]) 
                   
                   if reverse == False:
                   
                      tuple_list = (data.columns[i],data.columns[j])
                      
                   elif reverse == None:
                   
                      tuple_list = (data.columns[i],data.columns[j])   
                      
                   else:
                       
                      tuple_list = (data.columns[j],data.columns[i]) 
                      
                   in_stock = data.columns[j]
                   out_stock = data.columns[i]   
                   
                   score = 1
                   
                   pvalue = result_[1][0]['ssr_ftest'][1]
                   
                   Pvalue_list.append(pvalue)
                   
                   Score_list.append(score)
                   
                   Save_index.append(tuple_list)
                   
                   in_stock_list.append(in_stock)
                   out_stock_list.append(out_stock)
                   
                else :
                    
#                   print('{} does not granger cause {}'.format(data.columns[j],data.columns[i]))
                    
                   Check_list.append(['{} does not granger cause {}'.format(data.columns[j],data.columns[i])])
    
                   if reverse == False:
                   
                      tuple_list = (data.columns[i],data.columns[j])
                      
                   elif reverse == None:
                   
                      tuple_list = (data.columns[i],data.columns[j])   
                      
                   else:
                       
                      tuple_list = (data.columns[j],data.columns[i])
                      
                   in_stock = data.columns[j]
                   out_stock = data.columns[i]
                   
                   score = 0
                   
                   pvalue = result_[1][0]['ssr_ftest'][1]
                   
                   Pvalue_list.append(pvalue)
                   
                   Score_list.append(score)
                   
                   Save_index.append(tuple_list)    

                   in_stock_list.append(in_stock)
                   out_stock_list.append(out_stock)
         
    GC_indicator = pd.DataFrame(Check_list,columns=['GC result'],
                      index=pd.MultiIndex.from_tuples(Save_index))
    GC_indicator['P-value'] = Pvalue_list
    GC_indicator['GC score'] = Score_list
    GC_indicator['out stock'] = in_stock_list
    GC_indicator['in stock'] = out_stock_list
    
    return GC_indicator


#%%
subp1_GC = Grangercausality_test(subperiod1_data_,reverse = False)    
subp1_reverseGC = Grangercausality_test(subperiod1_data_,reverse = True)

subp2_GC = Grangercausality_test(subperiod2_data_,reverse = False)    
subp2_reverseGC = Grangercausality_test(subperiod2_data_,reverse = True)

subp3_GC = Grangercausality_test(subperiod3_data_,reverse = False)    
subp3_reverseGC = Grangercausality_test(subperiod3_data_,reverse = True)

subp4_GC = Grangercausality_test(subperiod4_data_,reverse = False)    
subp4_reverseGC = Grangercausality_test(subperiod4_data_,reverse = True)

subp5_GC = Grangercausality_test(subperiod5_data_,reverse = False)    
subp5_reverseGC = Grangercausality_test(subperiod5_data_,reverse = True)

subp6_GC = Grangercausality_test(subperiod6_data_,reverse = False)    
subp6_reverseGC = Grangercausality_test(subperiod6_data_,reverse = True)

subp1_total = pd.merge(subp1_GC, subp1_reverseGC, left_index=True, right_index=True, how='outer')
subp2_total = pd.merge(subp2_GC, subp2_reverseGC, left_index=True, right_index=True, how='outer')
subp3_total = pd.merge(subp3_GC, subp3_reverseGC, left_index=True, right_index=True, how='outer')
subp4_total = pd.merge(subp4_GC, subp4_reverseGC, left_index=True, right_index=True, how='outer')
subp5_total = pd.merge(subp5_GC, subp5_reverseGC, left_index=True, right_index=True, how='outer')
subp6_total = pd.merge(subp6_GC, subp6_reverseGC, left_index=True, right_index=True, how='outer')



#%% DGC for each sub periods

subp1_DGC = (subp1_total['GC score_x']+subp1_total['GC score_y']).sum(axis=0)/(len(subperiod1_data.columns)*(len(subperiod1_data.columns)-1))
subp2_DGC = (subp2_total['GC score_x']+subp2_total['GC score_y']).sum(axis=0)/(len(subperiod1_data.columns)*(len(subperiod1_data.columns)-1))
subp3_DGC = (subp3_total['GC score_x']+subp3_total['GC score_y']).sum(axis=0)/(len(subperiod1_data.columns)*(len(subperiod1_data.columns)-1))
subp4_DGC = (subp4_total['GC score_x']+subp4_total['GC score_y']).sum(axis=0)/(len(subperiod1_data.columns)*(len(subperiod1_data.columns)-1))
subp5_DGC = (subp5_total['GC score_x']+subp5_total['GC score_y']).sum(axis=0)/(len(subperiod1_data.columns)*(len(subperiod1_data.columns)-1))
subp6_DGC = (subp6_total['GC score_x']+subp6_total['GC score_y']).sum(axis=0)/(len(subperiod1_data.columns)*(len(subperiod1_data.columns)-1))

print("DGC of 2000.6.30 to 2001.12.31 is : ",100*round(subp1_DGC,2),"%")
print("DGC of 2002.1.01 to 2004.12.31 is : ",100*round(subp2_DGC,3),"%")
print("DGC of 2006.1.01 to 2008.12.31 is : ",100*round(subp3_DGC,3),"%")
print("DGC of 2009.1.01 to 2011.12.31 is : ",100*round(subp4_DGC,3),"%")
print("DGC of 2012.1.01 to 2015.12.31 is : ",100*round(subp5_DGC,4),"%")
print("DGC of 2016.1.01 to 2018.11.30 is : ",100*round(subp6_DGC,3),"%")

#%%
def Rolling_DGC(data,window):
    
    if len(data) < window:
               
       print("Observation window must be shorter than Data period(len(data))")
         
    rolling_DGCscore = []
    for j in range(len(data)-window):
                     
        data_ = data[j:j+window]
        
        GC_result = Grangercausality_test(data_,reverse = False)
        
        GC_resultrev = Grangercausality_test(data_,reverse = True)
        
        GC_scoresum = (GC_result['GC score']+GC_resultrev['GC score']).sum(axis=0)
        
        DGC = GC_scoresum/(len(data_.columns)*(len(data_.columns)-1))
        
        rolling_DGCscore.append(round(DGC,4))
            
    rolling_DGCscore = pd.DataFrame(rolling_DGCscore).set_index(data.index[window:])
    rolling_DGCscore.columns = ['rolling score']
    
    return rolling_DGCscore        

#rolling_DGC = Rolling_DGC(Data_total[:40],window=36)
#rolling_DGC2 = Rolling_DGC(Data_total[4:80],window=36)
#rolling_DGC3 = Rolling_DGC(Data_total[44:120],window=36)
#rolling_DGC4 = Rolling_DGC(Data_total[84:160],window=36)
#rolling_DGC5 = Rolling_DGC(Data_total[124:220],window=36)

#rolling_GDC_total = pd.concat([rolling_DGC,rolling_DGC2,rolling_DGC3,rolling_DGC3,rolling_DGC4,rolling_DGC5],axis=0)
#%%   
rolling_GDC_total = pd.read_csv('C:/Users/rbgud/OneDrive/바탕 화면/가을학기 후반/빅데이터와 금융자료분석/DGC_rolling graph data.csv',engine='python')
rolling_GDC_total.index = pd.to_datetime(rolling_GDC_total['DATE'],format='%Y-%m-%d')
rolling_GDC_total.pop('DATE')

plt.figure(figsize = (12,7)) 
plt.plot(rolling_GDC_total.iloc[:73,:].index,100*rolling_GDC_total.iloc[:73,:],label='Rolling DGC')
plt.plot(rolling_GDC_total.iloc[:73,:].index,np.ones(73)*6,'r-')
plt.plot(rolling_GDC_total.iloc[:73,:].index,np.ones(73)*5.5,'r--')
plt.plot(rolling_GDC_total.iloc[:73,:].index,np.ones(73)*5,'r-')
plt.ylabel('Rolling Degree of Granger Causality (%)')
plt.xlabel('Date')
plt.legend(loc='upper left')
plt.title('Rolling Degree of Granger Causality (window = 3yrs), 2003~2008')

plt.figure(figsize = (12,7)) 
plt.plot(rolling_GDC_total.index,100*rolling_GDC_total,label='Rolling DGC')
plt.plot(rolling_GDC_total.index,np.ones(len(rolling_GDC_total))*6,'r-')
plt.plot(rolling_GDC_total.index,np.ones(len(rolling_GDC_total))*5.5,'r--')
plt.plot(rolling_GDC_total.index,np.ones(len(rolling_GDC_total))*5,'r-')   
plt.ylabel('Rolling Degree of Granger Causality(%)')
plt.xlabel('Date')
plt.legend(loc='upper left')
plt.title('Rolling Degree of Granger Causality (window = 3yrs), Total Period')

plt.figure(figsize = (12,7)) 
plt.plot(rolling_GDC_total.index,100*rolling_GDC_total,label='Rolling DGC',linewidth=4)
plt.stackplot(x,y, labels=['PC1','PC5','PC10','PC25'], colors=pal, alpha=0.4 )
plt.ylabel('PCA and GC (%)')
plt.xlabel('Date')
plt.legend(loc='upper left')
plt.title('Compare PCA and Rolling Degree of Granger Causality (window = 3yrs), Total Period')

#%% Count number of connections 

def NumberofConnectionsforstock(data,in_out_flag,name_buff):
   
    dic={}
    
    ''' S(외부)에서 institutuion으로 들어오는 노드 개수'''
    df = Grangercausality_test(data,reverse = None)
    
    if in_out_flag == True :         # in case.. 들어오는 갯수 세기 number of in connections
    
       case1 = 'in stock'
       case2 = 'out stock'
    else:                            # out case.. 나가는 갯수 세기 number of out connections
       
       case1 = 'out stock' 
       case2 = 'in stock' 
    
    for i in name_buff:
        
        df_= df.loc[df[case1]== i]
        df2 = df_.loc[df_['GC score']==1]
        
        dic[i]=df2['GC score'].sum()
    
    result = pd.DataFrame([dic], index=['#ofConnections']).T
    
    return result

#%%
    
def NumberofConnectionsforsector(data,in_out_flag,name_buff):
   
    dic={}
    
    ''' S(외부)에서 institutuion으로 들어오는 노드 개수'''
    df = Grangercausality_test(data,reverse = None)
    
    if in_out_flag == True :         # in case.. 들어오는 갯수 세기 number of in connections
    
       case1 = 'in stock'
       case2 = 'out stock'
    else:                            # out case.. 나가는 갯수 세기 number of out connections
       
       case1 = 'out stock' 
       case2 = 'in stock' 
    
    for i in name_buff:
        
        df_= df.loc[df[case1]== i]
        df2 = df_.loc[df_['GC score']==1]
        
        dic[i]=df2['GC score'].sum()
    
    result = pd.DataFrame([dic], index=['#ofConnections']).T
    
    return result


#%%
def NumberofConnections(data,in_out_flag,sectorname,name_buff):
   
    hedge_nc=0
    bank_nc=0
    dealer_nc=0
    insur_nc=0
    
    ''' S(외부)에서 institutuion으로 들어오는 노드 개수'''
    df = Grangercausality_test(data,reverse = None)
    
    if in_out_flag == True :         # in case
    
       case1 = 'in stock'
       case2 = 'out stock'
    else:                            # out case
       
       case1 = 'out stock' 
       case2 = 'in stock' 
    
    for i in name_buff:
        
        df_= df.loc[df[case1]== i]
        df2 = df_.loc[df_['GC score']==1]
        
        for j in df2[case2].values:
            if j in bank_name_buff:
                bank_nc+=1
            elif j in dealer_name_buff:
                dealer_nc+=1
            elif j in hedge_name_buff:
                hedge_nc+=1    
            elif j in insurance_name_buff:
                insur_nc+=1
    result_df={'Hedge Funds':hedge_nc, 'Brokers': dealer_nc, 'Banks': bank_nc ,'Insurers': insur_nc}
    
    result = pd.DataFrame([result_df], index=[sectorname]).T
    
    return result

#%%
def NumberofSectorConnections(data,in_out_flag,sectorname,name_buff):
   
    hedge_nc=0
    bank_nc=0
    dealer_nc=0
    insur_nc=0
    
    ''' S(외부)에서 institutuion으로 들어오는 노드 개수'''
    df = Grangercausality_test(data,reverse = None)
    
    if in_out_flag == True :         # in case
    
       case1 = 'in stock'
       case2 = 'out stock'
    else:                            # out case
       
       case1 = 'out stock' 
       case2 = 'in stock' 
    
    for i in name_buff:
        
        df_= df.loc[df[case1]== i]
        df2 = df_.loc[df_['GC score']==1]
        
        for j in df2[case2].values:
            
            if name_buff == bank_name_buff:
                
               bank_nc += 0
            elif j in bank_name_buff:
               
               bank_nc += 1 
            
            if name_buff == dealer_name_buff:
                
               dealer_nc += 0
            elif j in dealer_name_buff:
               
               dealer_nc += 1 
            
            if name_buff == hedge_name_buff:
                
                hedge_nc = 0
    
            elif j in hedge_name_buff:
                hedge_nc+=1 
                
            if name_buff == insurance_name_buff: 
                
                insur_nc += 0
                
            elif j in insurance_name_buff:
                insur_nc+=1
                
    result_df={'Hedge Funds':hedge_nc, 'Brokers': dealer_nc, 'Banks': bank_nc ,'Insurers': insur_nc}
    
    result = pd.DataFrame([result_df], index=[sectorname]).T
    
    return result

#%% Graphing Network for each subperiod

subp1 = Grangercausality_test(subperiod1_data_,reverse=None)
subp1_ = subp1.loc[subp1['GC score']==1]
pair_list_subp1 = list(subp1_.index)

subp3 = Grangercausality_test(subperiod3_data_,reverse=None)
subp3_ = subp1.loc[subp3['GC score']==1]
pair_list_subp3 = list(subp3_.index)

#%%
def draw_graph(graph, labels=None, graph_layout='shell',
               node_size=1000, node_color='gray', node_alpha=0,
               node_text_size=7,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               text_font='sans-serif'):

    # create networkx graph
    G=nx.Graph()

    # add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])
        
    if graph_layout == 'spring':
        graph_pos=nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos=nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos=nx.random_layout(G)
    else:
        graph_pos=nx.shell_layout(G)
    
        

    # draw graph
    nx.draw_networkx_nodes(G,graph_pos,node_size=node_size, 
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G,graph_pos,width=edge_tickness,
                           alpha=edge_alpha,edge_color=edge_color)
    nx.draw_networkx_labels(G, graph_pos,font_size=node_text_size,
                            font_family=text_font)

    if labels is None:
        labels = range(len(graph))

    edge_labels = dict(zip(graph, labels))
    nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels, 
                                 label_pos=edge_text_pos)

    # show graph
   
    plt.show()
#%%
plt.figure(figsize = (15,9))    
plt.title('Network Diagram of Linear Granger causality relationships in 2000 to 2002 ')
draw_graph(pair_list_subp1)
#%%
plt.figure(figsize = (15,9))    
plt.title('Network Diagram of Linear Granger causality relationships in 2006 to 2008 ')
draw_graph(pair_list_subp3)

#%% Get Number of Connections result dataframes(in case)

#2000 to 2001
subp1_in_hedge = NumberofConnections(subperiod1_data_,in_out_flag=True,
                                     sectorname='Hedge Funds',name_buff=hedge_name_buff)
subp1_in_broker = NumberofConnections(subperiod1_data_,in_out_flag=True,
                                     sectorname='Brokers',name_buff=dealer_name_buff)
subp1_in_bank = NumberofConnections(subperiod1_data_,in_out_flag=True,
                                     sectorname='Banks',name_buff=bank_name_buff)               
subp1_in_insurance = NumberofConnections(subperiod1_data_,in_out_flag=True,
                                     sectorname='Insurers',name_buff=insurance_name_buff)

result_2000to2001 = pd.concat([subp1_in_hedge,subp1_in_broker,subp1_in_bank,subp1_in_insurance],axis=1)
result_2000to2001_pct = round(100*result_2000to2001/result_2000to2001.sum(axis=1).sum(axis=0))
Result_table_2000to2001 = pd.concat([result_2000to2001_pct,result_2000to2001],axis=1)

#2002 to 2004
subp2_in_hedge = NumberofConnections(subperiod2_data_,in_out_flag=True,
                                     sectorname='Hedge Funds',name_buff=hedge_name_buff)
subp2_in_broker = NumberofConnections(subperiod2_data_,in_out_flag=True,
                                     sectorname='Brokers',name_buff=dealer_name_buff)
subp2_in_bank = NumberofConnections(subperiod2_data_,in_out_flag=True,
                                     sectorname='Banks',name_buff=bank_name_buff)               
subp2_in_insurance = NumberofConnections(subperiod2_data_,in_out_flag=True,
                                     sectorname='Insurers',name_buff=insurance_name_buff)

result_2002to2004 = pd.concat([subp2_in_hedge,subp2_in_broker,subp2_in_bank,subp2_in_insurance],axis=1)
result_2002to2004_pct = round(100*result_2002to2004/result_2002to2004.sum(axis=1).sum(axis=0))
Result_table_2002to2004 = pd.concat([result_2002to2004_pct,result_2002to2004],axis=1)

#2006 to 2008
subp3_in_hedge = NumberofConnections(subperiod3_data,in_out_flag=True,
                                     sectorname='Hedge Funds',name_buff=hedge_name_buff)
subp3_in_broker = NumberofConnections(subperiod3_data,in_out_flag=True,
                                     sectorname='Brokers',name_buff=dealer_name_buff)
subp3_in_bank = NumberofConnections(subperiod3_data,in_out_flag=True,
                                     sectorname='Banks',name_buff=bank_name_buff)               
subp3_in_insurance = NumberofConnections(subperiod3_data,in_out_flag=True,
                                     sectorname='Insurers',name_buff=insurance_name_buff)

result_2006to2008 = pd.concat([subp3_in_hedge,subp3_in_broker,subp3_in_bank,subp3_in_insurance],axis=1)
result_2006to2008_pct = round(100*result_2006to2008/result_2006to2008.sum(axis=1).sum(axis=0))
Result_table_2006to2008 = pd.concat([result_2006to2008_pct,result_2006to2008],axis=1)

#2009 to 2011
subp4_in_hedge = NumberofConnections(subperiod4_data_,in_out_flag=True,
                                     sectorname='Hedge Funds',name_buff=hedge_name_buff)
subp4_in_broker = NumberofConnections(subperiod4_data_,in_out_flag=True,
                                     sectorname='Brokers',name_buff=dealer_name_buff)
subp4_in_bank = NumberofConnections(subperiod4_data_,in_out_flag=True,
                                     sectorname='Banks',name_buff=bank_name_buff)               
subp4_in_insurance = NumberofConnections(subperiod4_data_,in_out_flag=True,
                                     sectorname='Insurers',name_buff=insurance_name_buff)

result_2010to2012 = pd.concat([subp4_in_hedge,subp4_in_broker,subp4_in_bank,subp4_in_insurance],axis=1)
result_2010to2012_pct = round(100*result_2010to2012/result_2010to2012.sum(axis=1).sum(axis=0))
Result_table_2010to2012 = pd.concat([result_2010to2012_pct,result_2010to2012],axis=1)

#2012 to 2015
subp5_in_hedge = NumberofConnections(subperiod5_data_,in_out_flag=True,
                                     sectorname='Hedge Funds',name_buff=hedge_name_buff)
subp5_in_broker = NumberofConnections(subperiod5_data_,in_out_flag=True,
                                     sectorname='Brokers',name_buff=dealer_name_buff)
subp5_in_bank = NumberofConnections(subperiod5_data_,in_out_flag=True,
                                     sectorname='Banks',name_buff=bank_name_buff)               
subp5_in_insurance = NumberofConnections(subperiod5_data_,in_out_flag=True,
                                     sectorname='Insurers',name_buff=insurance_name_buff)

result_2013to2015 = pd.concat([subp5_in_hedge,subp5_in_broker,subp5_in_bank,subp5_in_insurance],axis=1)
result_2013to2015_pct = round(100*result_2013to2015/result_2013to2015.sum(axis=1).sum(axis=0))
Result_table_2013to2015 = pd.concat([result_2013to2015_pct,result_2013to2015],axis=1)

#2016 to 2018
subp6_in_hedge = NumberofConnections(subperiod6_data_,in_out_flag=True,
                                     sectorname='Hedge Funds',name_buff=hedge_name_buff)
subp6_in_broker = NumberofConnections(subperiod6_data_,in_out_flag=True,
                                     sectorname='Brokers',name_buff=dealer_name_buff)
subp6_in_bank = NumberofConnections(subperiod6_data_,in_out_flag=True,
                                     sectorname='Banks',name_buff=bank_name_buff)               
subp6_in_insurance = NumberofConnections(subperiod6_data_,in_out_flag=True,
                                     sectorname='Insurers',name_buff=insurance_name_buff)

result_2016to2018 = pd.concat([subp6_in_hedge,subp6_in_broker,subp6_in_bank,subp6_in_insurance],axis=1)
result_2016to2018_pct = round(100*result_2016to2018/result_2016to2018.sum(axis=1).sum(axis=0))
Result_table_2016to2018 = pd.concat([result_2016to2018_pct,result_2016to2018],axis=1)

Result_Total = pd.concat([Result_table_2000to2001,Result_table_2002to2004,
                          Result_table_2006to2008,Result_table_2010to2012,
                          Result_table_2013to2015,Result_table_2016to2018],axis=0)

Result_Total.to_csv('C:/Users/rbgud/OneDrive/바탕 화면/가을학기 후반/빅데이터와 금융자료분석/Result_Total3.csv')


#%%

in_connection=NumberofConnectionsforstock(Data_total['2002-10':'2005-09'],
                                          in_out_flag=True, name_buff=name_buff)

out_connection=NumberofConnectionsforstock(Data_total['2002-10':'2005-09'], 
                                           in_out_flag=False, name_buff=name_buff)

in_connection_=NumberofConnectionsforstock(Data_total['2004-07':'2007-06'],
                                          in_out_flag=True, name_buff=name_buff)

out_connection_=NumberofConnectionsforstock(Data_total['2004-07':'2007-06'], 
                                           in_out_flag=False, name_buff=name_buff)

dic={}
def reg_m_(y, x):
    empty=pd.concat([y,x], axis=1)
    ones = np.ones(len(x))
    X = sm.add_constant(np.column_stack((empty['#ofConnections'], ones)))
    Y=np.array(empty['Max%LOSS'].values).reshape(len(y),1)
    results=sm.OLS(Y, X).fit()
    tau, _ = stats.kendalltau(x,y) 
    return results.params[0], results.tvalues[0], results.pvalues[0], tau

in_connection_rank=in_connection.rank(ascending=False)
out_connection_rank=out_connection.rank(ascending=False)
in_connection_rank_=in_connection_.rank(ascending=False)
out_connection_rank_=out_connection_.rank(ascending=False)
#in_out_connection_rank = in_out_connection.rank(ascending=False)



dic['# of "In" Connections']=reg_m_(percent_loss, in_connection_rank)
dic['# of "Out" Connections']=reg_m_(percent_loss, out_connection_rank)
dic['# of "In" Connections_']=reg_m_(percent_loss, in_connection_rank_)
dic['# of "Out" Connections_']=reg_m_(percent_loss, out_connection_rank_)

reg_result=pd.DataFrame.from_dict(dic,orient='index', columns=['Coeff', 't-stat', 'p-value', 'Kendall t'])