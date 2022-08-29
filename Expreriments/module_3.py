import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from matplotlib import pyplot as plt
import ssl
import warnings
warnings.filterwarnings("ignore")
import riskfolio as rp
import time

# set the max columns to none
pd.set_option('display.max_columns', None)

# Inputs 

bse_ticker = pd.read_csv('Master.csv')
bse_ticker.set_index('NSE Ticker',inplace=True)
cd = pd.read_csv('closing_dataset.csv',index_col='Date',parse_dates=True)

# Defining the Parameters for experiments
para = {
'OPM':[True,5,'>'],
'CFO':[True,0,'>'],
'Net Cash Flow':[True,0,'>'],
'Debt/Asset':[True,0.4,'<'],
'Other Income/Total Revenue':[True,0.10,'<'],
'Other Liabilities/Total Liabilities':[False,0.30,'<'],
'Investment/Total Assets':[True,0.20,'<']
}

para_s = {
    'Financing Margin %':[True,12,'>'],
    'Net_Profit':[True,0,'>'],
    'Net Cash Flow':[False,0,'>']
}

# Get the Closing Price Dataframe 

def closing_dataset_funtion(tickers,start,end):
    df = yf.download(tickers, start=start, end=end,progress=False,show_errors=False)['Close']
    return df

def RSI_dataset_function(dataset,n2=14):
    
    close_price = dataset.copy()
    
    for i in close_price:
        a = pd.DataFrame(close_price[i])
        
        # Omtting null values
        a = a.dropna(axis=0)
        
        change = a.diff().dropna()
        change
        
        change_up = change.clip(lower=0)
        
        change_down = (-1 * change.clip(upper=0)).abs()
        

        avg_up = change_up.rolling(n2).mean()
    
        avg_down = change_down.rolling(n2).mean().abs()

        rs = avg_up/avg_down   # calculating rs
        
        rsi = 100-(100/(1+rs))  # calculating RSI
        rsi.dropna(inplace=True)
        close_price[i] = rsi
    
    return close_price

def SMA_dataset_function(dataset,n=5):
    close_price = dataset.copy()
    return close_price.rolling(n).mean()

def regr(x,y):
    regr1 = LinearRegression()
    regr1.fit(x,y)
    return regr1.coef_


def get_Slope(dataset): ## Input would be the Dataset
    close_price = dataset.copy()
    index_list = []
    regr2 = []   ### Empty list to store the slope of the input dataset.
    y_hat = close_price ### Defining y_hat which is same as dataset.
    y_hat = y_hat.reset_index(drop=True)    ### Resetting the index so that date becomes a column.

    for i in y_hat.columns[1:]:   #### Iterating through the columns of y_hat.
        y = y_hat[i]   ### y is taken as the each data point of each and every column.
        y.dropna(inplace=True)
        if len(y)>0:
            x = np.array(y.index).reshape(-1, 1)   ##The x axis taken as sequence of numbers. 
            regr2.append(regr(x,y))   ## Appending the coefficients to the empty list.
            index_list.append(i)
    return pd.DataFrame(regr2, index = index_list)   ### Return the Dataframe and the index is set as column names.

def RSI_divergence_function(SMA_dataset,RSI_dataset):
    SMA_slope = get_Slope(SMA_dataset)
    SMA_slope.rename(columns={0:'SMA'},inplace=True)
    RSI_slope = get_Slope(RSI_dataset)
    RSI_slope.rename(columns={0:'RSI'},inplace=True)
    divergence = pd.concat([SMA_slope,RSI_slope],axis=1)
    divergence['divergence'] = True

    for i in range(len(divergence)):
        if not (divergence.iloc[i][0] < 0) & (divergence.iloc[i][1]>0):
            divergence.divergence.iloc[i] = False
    return divergence

def RSI_breakout_function(new_RSI_dataset,breakout_RSI_dataset): 
    """
    input: two RSI dataset which is broken in two windows
    """
     # taking the last value of both the windows
    new_RSI_dataset = new_RSI_dataset[-1:]
    breakout_RSI_dataset = breakout_RSI_dataset[-1:]
    
    # checking the last value of the last window is lesser than the latest window
    RSI_breakout = pd.DataFrame(new_RSI_dataset.iloc[0] < breakout_RSI_dataset.iloc[0])
    
    # changing the name of the column
    RSI_breakout.rename(columns={0:'RSI_breakout'},inplace=True)
    
    # returning RSI_breakout
    return RSI_breakout

def breakout_function(dataset,breakout_window = 10,RSI_divergence_window = 30):
    close_price = dataset.copy()
#     print(dataset.shape)
    
    # tries to two dataset which is RSI and SMA
    RSI_dataset = RSI_dataset_function(close_price)
    SMA_dataset = SMA_dataset_function(close_price)
#     print(RSI_dataset.shape)
#     print(SMA_dataset.shape)
    
    
    # Breaking the Dataset into Two Windows 
    
    # RSI DIVERGENCE WINDOW
    new_RSI_dataset = RSI_dataset[-RSI_divergence_window-breakout_window:-breakout_window]
    new_SMA_dataset = SMA_dataset[-RSI_divergence_window-breakout_window:-breakout_window]
#     print(new_RSI_dataset.shape)
#     print(new_SMA_dataset.shape)

    # BREAKOUT WINDOW
    breakout_RSI_dataset = RSI_dataset[-breakout_window:]
    breakout_SMA_dataset = SMA_dataset[-breakout_window:]
#     print(breakout_RSI_dataset.shape)
#     print(breakout_SMA_dataset.shape)
    
    # Calculating RSI Divergence
    RSI_divergence = RSI_divergence_function(new_SMA_dataset,new_RSI_dataset)
    
    # Calculating Price Breakout
#     price_breakout = price_breakout_function(new_SMA_dataset,breakout_SMA_dataset)

    # Calculating RSI Breakout
#     RSI_breakout = RSI_breakout_function(new_RSI_dataset,breakout_RSI_dataset)

    
    # Calculating Breakout SMA Slope
    breakout_slope = get_Slope(breakout_SMA_dataset)
    breakout_slope.rename(columns={0:'SMA_breakout'},inplace=True)
    
    # Calculating Breakout SMA Slope
    breakout_RSI_slope = get_Slope(breakout_RSI_dataset)
    breakout_RSI_slope.rename(columns={0:'RSI_breakout'},inplace=True)

    
    # Concatenating all the Dataframes
    breakout = pd.concat([RSI_divergence,breakout_slope,breakout_RSI_slope],axis=1)
    
    # Adding a breakout column with 
    breakout['breakout'] = None
    
#     breakout
    for i in range(len(breakout)):
        if ((breakout.iloc[i][2] == True)and(breakout.iloc[i][3]>0)and breakout.iloc[i][4]>0):
#         if ((breakout.iloc[i][2] == True)and(breakout.iloc[i][3]>0)):            
#             print(i)
#             print(True)
            breakout.breakout[i] = True
    return breakout

# Screener.in Scrapping

def fs_concat(pnl,bs,cf):
    """
    This function takes in the pnl,bs,cf and concatenate them
    """
    # concatenating the tables 
    fs = pd.concat([pnl,bs,cf])
    
    # setting the primary column as index 
    fs.set_index('Unnamed: 0',inplace=True)
    fs = fs.T
    return fs 
# fs_concat(pnl,bs,cf)

def cagr_function(csg,cpg,spc,roe):
    """
    This function takes in the csg, cpg, spc and roe table and concatenate them
    """
    
    # setting primary columns as index
    csg.set_index('Compounded Sales Growth',inplace=True)
    cpg.set_index("Compounded Profit Growth",inplace=True)
    spc.set_index("Stock Price CAGR",inplace=True)
    roe.set_index("Return on Equity",inplace=True)

    # concatenating
    cagr = pd.concat([csg,cpg,spc,roe],axis=1)

    return cagr
# cagr_function(csg,cpg,spc,roe)

def screener_scrape_function(ticker):
    """
    This function takes in the BSE ticker for a company and returns a financial statements(pnl,bs,cf), CAGR and ratios for the respectice company
    1. takes in the tickers
    2. add the ticker to the screener.in url
    3. extract the tables from above tickers
    4. assign variable to the different tables
    5. concatenate fs,pnl,cf table into single fs (financial statement) table
    6. concatenate csg,cpg,spc,roe in cage table
    """
    
    # Scrapping the tables from the url
    ssl._create_default_https_context = ssl._create_unverified_context
    
    scraped = pd.read_html(f"https://www.screener.in/company/{ticker}")
    
    # assinging variables to different tables
    qr = scraped[0] # Quaterly results
    pnl = scraped[1] # profit and loss
    csg = scraped[2] # compounded sales growth
    cpg = scraped[3] # compunded profit growth 
    spc = scraped[4] # stock price CAGR
    roe = scraped[5] # return on Equity
    bs = scraped[6] # balance sheet 
    cf = scraped[7] # cash flow 
    ratios = scraped[8] # ratios 
#     shareholding = scraped[9] # shareholding pattern
    
    # concatenating pbl,bs,cf
    fs = fs_concat(pnl,bs,cf)
    
    # concatenating csg,cpg,spc,roe
    cagr = cagr_function(csg,cpg,spc,roe)
    
    # setting the first columns as index in ratios
    ratios.set_index('Unnamed: 0',inplace=True)
    
    return fs,cagr,ratios.T
# fs,cagr,ratios = screener_scrape_function(541988)

def screener_data_function(lst,bse_list):
    fs_final = pd.DataFrame()
    cagr_final = pd.DataFrame()
    ratios_final = pd.DataFrame()

    for i in range(len(bse_list)):
#         print(bse_list[i])
        if (bse_list[i] != 'BSE') &  (bse_list[i] != 'CDSL'):
            time.sleep(1.5)
            fs,cagr,ratios = screener_scrape_function(int(bse_list[i]))
            fs = fs.reset_index()
            fs['Company'] = lst[i]
            fs_final = pd.concat([fs_final,fs])

            cagr = cagr.reset_index()
            cagr['Company'] = lst[i]
            cagr_final = pd.concat([cagr_final,cagr])

            ratios = ratios.reset_index()
            ratios['Company'] = lst[i]
            ratios_final = pd.concat([ratios_final,ratios])
        else:
            time.sleep(1.5)
            fs,cagr,ratios = screener_scrape_function(str(bse_list[i])+'/consolidated/')
            fs = fs.reset_index()
            fs['Company'] = lst[i]
            fs_final = pd.concat([fs_final,fs])

            cagr = cagr.reset_index()
            cagr['Company'] = lst[i]
            cagr_final = pd.concat([cagr_final,cagr])

            ratios = ratios.reset_index()
            ratios['Company'] = lst[i]
            ratios_final = pd.concat([ratios_final,ratios])
            
        
    return fs_final,cagr_final,ratios_final
    

def rename_columns(fs_final,cagr_final,ratios_final):
    fs_rename ={
        'index':'Year',
        'OPM %':'OPM',
        'Cash from Operating Activity\xa0+':'CFO',
        'Cash from Investing Activity\xa0+':'CFI',
        'Cash from Financing Activity\xa0+':'CFF',
        'Other Assets\xa0+':'Other_Assets',
        'Sales\xa0+':'Sales',
        'Expenses\xa0+':'Expenses',
        'Other Income\xa0+':'Other_Income',
        'Share Capital\xa0+':'Share_Capital',
        'Other Liabilities\xa0+':'Other_Liabilities',
        'Fixed Assets\xa0+':'Fixed_Assets',
        'Net Profit': 'Net_Profit'
    }

    fs_final.rename(columns=fs_rename,inplace=True)

    cagr_rename = {
        'index':'Year',
        'Compounded Sales Growth.1':'Sales Growth',
        'Compounded Profit Growth.1':'Profit Growth',
        'Stock Price CAGR.1':'Stock Price Growth',
        'Return on Equity.1':'Return on Equity.1'

    }

    cagr_final.rename(columns=cagr_rename, inplace=True)

    ratios_rename = {
        'index':'Year'
    }

    ratios_final.rename(columns=ratios_rename,inplace=True)
# rename_columns()

def remove_percentage_function(check_df):
    percentage_list = ['Tax %','OPM','Dividend Payout %','Financing Margin %']
    for column in percentage_list:
        try:
#             print(column)
            for i in range(len(check_df[column])):


                if type(check_df[column].iloc[i]) == str:
                    if ',' in check_df[column].iloc[i]:
                        check_df[column].iloc[i] = check_df[column].iloc[i].replace(',','')
                    check_df[column].iloc[i] = check_df[column].iloc[i][:-1]
                else:
                    check_df[column].iloc[i] = 0 
        except:
            None
# remove_percentage_function(check_df)

def column_type_convertor(check_df):
    for i in check_df:
        if (i != 'Year') & (i!='Company') & (i!='Industry'):
            check_df[i] = pd.to_numeric(check_df[i])
# column_type_convertor(check_df)

    # para = {
    #     'OPM':[True,12,'>'],
    #     'CFO':[True,0,'>'],
    #     'Net Cash Flow':[True,0,'>']
    # }
    # para_s = {
    #     'Financing Margin %':[True,12,'>'],
    #     'Net_Profit':[True,0,'>'],
    #     'Net Cash Flow':[False,0,'>']
    # }
def filter_stocks(para,para_s,check_df):
    df = check_df.copy()
    df = df[df['Industry']!= 'Financial Services']

    # Financial Service Comapny
    df_s = check_df.copy()
    df_s = df_s[df_s['Industry']== 'Financial Services']
    df_s

    filters = df.copy()
    filters_s = df_s.copy()


    for i in para:
        if para[i][0]:
            if para[i][2] == '>':
                filters = df[(df[i]>=para[i][1])]
            elif para[i][2] == '<':
                filters = df[(df[i]<=para[i][1])]
            else:
                filters = df[(df[i]==para[i][1])]
        df = filters


    if df_s.shape[0]>0:
        try:
            for i in para_s:
                if para_s[i][0]:
                    if para_s[i][2] == '>':
                        filters_s = df_s[(df_s[i]>=para_s[i][1])]
                    elif para[i][2] == '<':
                        filters_s = df_s[(df_s[i]<=para_s[i][1])]
                    else:
                        filters_s = df_s[(df_s[i]==para_s[i][1])]
                df_s = filters_s
        except:
            for i in para:
                if para[i][2] == '>':
                    filters_s = df_s[(df_s[i]>=para[i][1])]
                elif para[i][2] == '<':
                    filters_s = df_s[(df_s[i]<=para[i][1])]
                else:
                    filters_s = df_s[(df_s[i]==para[i][1])]
                df_s = filters_s
#     display(df)
#     display(df_s)
#     display(check_df,df_s,df)
    filtered_ticker = list(df['Company'].values) + list(df_s['Company'].values)
    return filtered_ticker

# year_1 = 'Dec 2019'
# year_2 = 'Mar 2020'
# check_df = fs_final[(fs_final['Year']==year_1) | (fs_final['Year']==year_2)]

# filtered_ticker = list(check_df[(check_df['CFO']>0)]['Company'].values)

# Get The Returns Dataframe 

def returns_dataset_funtion(filtered_ticker,closing_dataset):
    return_df = pd.DataFrame() # Creating a empty Df
    close_price = closing_dataset.copy() # Creating a copy of the closing dataset
#     close_price.reset_index(inplace=True)

    # for every company in filtered list. 
    for i in filtered_ticker:
        a = close_price[i] # get a column from the closing price dataset
        # Omtting null values
        a = a.dropna(axis=0) # omit all the nan values
        a = a.pct_change().dropna() # get a return and remove the first columns
        return_df[i] = a

    return return_df
# returns_dataset = returns_dataset_funtion(filtered_ticker,closing_dataset)

# HRP function

def weightshrp(df1) :
    port = rp.HCPortfolio(returns=df1)
    model='HRP'
    codependence = 'pearson'
    rm = 'MV'
    rf = 0
    linkage = 'single'
    max_k = 10
    leaf_order = True
    w = port.optimization(model=model,
                          codependence=codependence,
                          rm=rm,
                          rf=rf,
                          linkage=linkage,
                          max_k=max_k,
                          leaf_order=leaf_order)
    return(w)
# weightshrp(returns_dataset)

# Part 1  

# start = '2020-04-30'
# end = '2021-04-30'
# bse_ticker = pd.read_csv('Master.csv')
# bse_ticker.set_index('NSE Ticker',inplace=True)
# cd = pd.read_csv('closing_dataset.csv',index_col='Date',parse_dates=True)

# breakout_window=10
# RSI_divergence_window = 30


def part_1(dataset,bse_ticker,start,end,breakout_window=10,RSI_divergence_window=40):
    start_index = pd.date_range(start=start,end=end)
#     future_index = pd.date_range(start=future_start,end=future_end)
 
    # Part 1
    closing_dataset = dataset.loc[dataset.index.intersection(start_index)]
    
    RSI_dataset = RSI_dataset_function(closing_dataset)
    SMA_dataset = SMA_dataset_function(closing_dataset)
#     print("Part 1 Done")
    
    # Part 2 
    a = breakout_function(closing_dataset,breakout_window=breakout_window,RSI_divergence_window=RSI_divergence_window)
    
    breakout = a.loc[a['breakout']==True]
    z = list(breakout.index)
    breakout['Industry'] = bse_ticker['Industry'].loc[z].values
#     print(breakout.index)
#     display(breakout) ######
    lst = list(breakout.index)
#     print(lst)
    b = list(bse_ticker.loc[lst]['BSE Ticker'].values)
#     print(b)
    return lst,b,closing_dataset
    

# lst,b,closing_dataset = part_1(cd,bse_ticker,start=start,end=end,breakout_window=breakout_window,RSI_divergence_window=RSI_divergence_window)

# Part 2 

# year_1 = 'Dec 2020'
# year_2 = 'Mar 2021'

# para = {
# 'OPM':[False,12,'>'],
# 'CFO':[True,0,'>'],
# 'Net Cash Flow':[False,0,'>']
#     }

# para_s = {
#     'Financing Margin %':[True,12,'>'],
#     'Net_Profit':[True,0,'>'],
#     'Net Cash Flow':[False,0,'>']
# }


def part_2(lst,b,year_1,year_2,para,para_s):
    fs_final,cagr_final,ratios_final = screener_data_function(lst,b)
    rename_columns(fs_final,cagr_final,ratios_final)

    check_df = fs_final[(fs_final['Year']==year_1) | (fs_final['Year']==year_2)]
    
    z = list(check_df['Company'].values)
    check_df['Industry'] = bse_ticker['Industry'].loc[z].values

    
    # Remove Percentage 
    remove_percentage_function(check_df)
    # Convert Columns to Integers
    column_type_convertor(check_df)
    check_df['Debt/Asset'] = (check_df['Borrowings']+check_df['Other_Liabilities'])/(check_df['Total Assets'])
    check_df['Other Income/Total Revenue'] = check_df['Other_Income']/check_df['Sales']
    check_df['Other Liabilities/Total Liabilities'] = check_df['Other_Liabilities']/check_df['Total Liabilities']
    check_df['Investment/Total Assets'] = check_df['Investments']/check_df['Total Assets']
    
    
    filtered_ticker = filter_stocks(para,para_s,check_df)
    
    


#     filterd = check_df[(check_df['CFO']>CFO_amount)]
#     display(filterd)
#     filtered_ticker = list(filterd['Company'].values)
#     display(check_df)
#     print(filtered_ticker)
    return filtered_ticker

# filtered_ticker = part_2(lst,b,year_1,year_2,para,para_s)
# filtered_ticker

# Part 4

def part_4(filtered_ticker,closing_dataset):
    if len(filtered_ticker)>1:
        returns_dataset = returns_dataset_funtion(filtered_ticker,closing_dataset)
        w = weightshrp(returns_dataset.dropna())
#         display(w)
    else:
        print('Only 1 Company Found')
        return pd.DataFrame()
    return w

# w = part_4(filtered_ticker,closing_dataset)
# w

# Part 5

# future_start = '2021-04-30'
# future_end= '2021-05-31'
# future_index = pd.date_range(start=future_start,end=future_end)


def part_5(dataset,future_start,future_end,w,filtered_ticker,m):
#     display(w)
    if w.shape[0] > 0:
        future_index = pd.date_range(start=future_start,end=future_end)
        future_price_dataset = dataset.loc[dataset.index.intersection(future_index)]
        future_price = future_price_dataset.get(filtered_ticker)
        po = np.asarray(future_price[:1])
        p1 = np.asarray(future_price[-1:])
#         m = 100000
        port = m * np.array(w)
        re = p1/po
        print(np.sum(re * np.asarray(port).T))
        return np.sum(re * np.asarray(port).T)
    else:
        print('Only 1 Company Found')
        return m
    
    # main(cd,bse_ticker,start=start,end=end,year_1=year_1,year_2=year_2,future_start=future_start,future_end=future_end,m=100000)

# part_5(cd,future_start,future_end,w)


def main(start,end,future_start,future_end,m,year_1,year_2,breakout_window=10,RSI_divergence_window=30):
    start = start
    end = end


    lst,b,closing_dataset = part_1(cd,bse_ticker,start=start,end=end,breakout_window=breakout_window,RSI_divergence_window=RSI_divergence_window)
    RSI_list = len(lst)
    print('Total companies which shows RSI Divergence',RSI_list)

    if len(lst) == 0:
        print("RSI NOT FOUND")
    else:
        filtered_ticker = part_2(lst,b,year_1,year_2,para,para_s)
        fundamental_list = len(filtered_ticker)
        print('Total Companies which passed fundamental filter',fundamental_list)

        w = part_4(filtered_ticker,closing_dataset)
#         display(w)
        print('Total Companies went into HRP model',w.shape[0])

        future_start = future_start
        future_end= future_end
        # future_index = pd.date_range(start=future_start,end=future_end)
        money = part_5(cd,future_start,future_end,w,filtered_ticker,m)
    try:
        return money,RSI_list,fundamental_list
    except:
        return m
    
