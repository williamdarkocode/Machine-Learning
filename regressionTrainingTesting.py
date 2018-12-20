import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression
# prerprocessing allows us to scale data; goal is to have features between -1 and 1
# cross validation helps with shuffling data
#svm stands for support vector machine


#df stands for data-frame; can be any variable name; df is for convention
df = quandl.get('WIKI/GOOGL')


# print(df.head())



# make new column HI_OP of elements in High col minus elements in Open col
# open is the start price of stock
#volume is how many trades happened that day

df['TOTAL_CHNG'] = df['Adj. Close'] - df['Adj. Open']
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100

df['PCT_CHNG'] =  (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Close'] * 100

df = df[['Adj. Close','HL_PCT','PCT_CHNG', 'TOTAL_CHNG']]
print(df.tail())

forecast_col = 'Adj. Close'

# fill in nan elements.
df.fillna(-99999, inplace=True)


# predict out ten percent of data frame, using math.ceil in case decimal
forecast_out = int(math.ceil(0.01*len(df)))
# number of days into future we're predicting
print(forecast_out)

# labels are predictions (forcasted values) of forcasted column
df['Label'] = df[forecast_col].shift(-forecast_out)

# drop nan values
df.dropna(inplace=True)

# print end of data instead of start
# print(df.head())

# features; drop label column becase features are evrything but label column
#df.drop() returns new dataframe; X is new dataframe of all features
X = np.array(df.drop(['Label'],1))

# lables
Y = np.array(df['Label'])

# scale X, the features, before giving to classifier
X = preprocessing.scale(X)

y = np.array(df['Label'])


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# classifier
# runs linearly, one job after the other.
# clf = LinearRegression()
# run 10 threads/jobs at a time, instead of lineraly; inheretly faster
clf = LinearRegression(n_jobs=10)
# run as many jobs/threads as possible... rip your processor lmao
# clf = LinearRegression(n_jobs=-1)


# support vector regression instead of linear
# clf = svm.SVR()
# svm regression but running linearly
# clf = svm.SVR(kernel='poly')

# train classifier
clf.fit(X_train, y_train)
# test classifier
accuracy = clf.score(X_test, y_test)
# accuracy in linear regression is going to be the squared error
# accuracy of what the price will be 1 percent of the days into the future
print(accuracy)

print(df.tail())





