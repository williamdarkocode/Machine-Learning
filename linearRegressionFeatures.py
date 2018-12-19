import pandas as pd
import quandl



#df stands for data-frame; can be any variable name; df is for convention
df = quandl.get('WIKI/GOOGL')


print(df.head())

df = df[['Adj. Close', 'Adj. Volume', 'High', 'Open',]]


# make new column HI_OP of elements in High col minus elements in Open col
# open is the start price of stock
#volume is how many trades happened that day

df['TOTAL_OP_CL_DIFF'] = df['Open'] - df['Adj. Close']
df['HL_PCT'] = (df['High'] - df['Adj. Close']) / 100

df['PCT_CHNG'] =  (df['High'] - df['Adj. Close']) / df['Adj. Close'] * 100

df = df[['Adj. Close','HL_PCT','PCT_CHNG', 'TOTAL_OP_CL_DIFF']]


print(df.head())
