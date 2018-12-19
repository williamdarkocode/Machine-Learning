import pandas as pd
import quandl



#df stands for data-frame; can be any variable name; df is for convention
df = quandl.get('WIKI/GOOGL')


print(df.head())

df = df[['Adj. Close', 'Adj. Volume', 'High', 'Open',]]


# make new column HI_OP of elements in High col minus elements in Open col
df['HI_OP'] = (df['High'] - df['Open'])


