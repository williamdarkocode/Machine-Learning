import pandas as pd
import quandl
import math


#df stands for data-frame; can be any variable name; df is for convention
df = quandl.get('WIKI/GOOGL')


print(df.head())



# make new column HI_OP of elements in High col minus elements in Open col
# open is the start price of stock
#volume is how many trades happened that day

df['TOTAL_CHNG'] = df['Adj. Close'] - df['Adj. Open']
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100

df['PCT_CHNG'] =  (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Close'] * 100

df = df[['Adj. Close','HL_PCT','PCT_CHNG', 'TOTAL_CHNG']]


forecast_col = 'Adj. Close'

# fill in nan elements.
df.fillna(-99999, inplace=True)


# predict out ten percent of data frame, using math.ceil in case decimal
forecast_out = int(math.ceil(0.01*len(df)))

df['Label'] = df[forecast_col].shift(-forecast_out)

# drop nan values
df.dropna(inplace=True)

# print end of data instead of start
print(df.head())



# print(df.head())
