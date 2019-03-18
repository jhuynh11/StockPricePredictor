import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import style
import datetime


def plotting():
    style.use('ggplot')

    # Read the data from csv file
    df = pd.read_csv('../Consolidated_Data/Apple.csv')

    df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume', 'Sentiment']]
    forecast_col = 'Close'
    df.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(0.01 * len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out)

    X = np.array(df.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    df.dropna(inplace=True)

    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = svm.SVR(kernel="rbf")
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(confidence)

    forecast_set = clf.predict(X_lately)
    df['Forecast'] = np.nan


    # df['Close'].plot()
    df['Close'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


def make_model(company, model, num_days, sentiment):

    df = pd.read_csv('../Consolidated_Data/' + company + '.csv')
    company_prices = list(df['Close'])

    # Create feature vectors
    df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    if sentiment:
        df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume', 'Sentiment']]
    else:
        df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]

    X = np.array(df)

    Y = []
    for i in range(num_days, len(df)- num_days):
        Y.append(1 if company_prices[i+num_days] > company_prices[i] else -1)
    print(len(Y))

    # Split training and testing sets
    X_train = np.array(X[0:])

    # clf = svm.SVC(kernel='rbf')
    # clf.fit(X_train, y_train)
    # score = clf.score(X_test, y_test)
    # print(score)

make_model("Apple", "", 1, False)