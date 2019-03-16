import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Read the data from csv file
df = pd.read_csv('../Historical_Data/AAPL.csv')
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

# Percent Change from high to low
df['HL_PCT'] = (df['High'] - df['Low']) / df['Low'] * 100.0


df['PCT_Change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Close', 'HL_PCT', 'PCT_Change', 'Volume']]

forecast_col = 'Close'

# Fill non-applicable values e.g blanks. In machine learning, we can't work with NaN
# thus we must replace it with something usable.
df.fillna(-9999, inplace=True)

# forecast_out = int(math.ceil(0.1*len(df))) # 10% of the dataframe
forecast_out = 30
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
Y = np.array(df['label'])

# Want features to be between -1 to 1 to improve speed
X = preprocessing.scale(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35)

clf = LinearRegression()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print(accuracy)