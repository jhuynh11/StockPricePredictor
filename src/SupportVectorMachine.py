import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
from statistics import mean, median


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


def get_stock_momentum(num_days, closing_prices):
    # Average of a stock's momentum over the past num_days. Each day is labeled 1 if
    # the closing price that day is higher than the closing price of the day before, and -1 if its lower
    momentum = []
    stock_momentum = []

    for i in range(num_days, len(closing_prices)):
        momentum.append(1 if closing_prices[i] > closing_prices[i - 1] else -1)

    for i in range(num_days, len(closing_prices)):
        stock_momentum.append(mean(momentum[i - num_days:i]))

    return stock_momentum


def get_volatility(num_days, closing_prices):
    # Stock price volatility. This is an average over the past num_days of
    # percent change in a stock's price per day
    volatility = []
    avg_volatility = []

    for i in range(num_days, len(closing_prices)):
        volatility.append((closing_prices[i] - closing_prices[i-1])/closing_prices[i-1])

    for i in range(num_days, len(closing_prices)):
        avg_volatility.append(mean(volatility[i - num_days:i]))

    return avg_volatility


def get_sentiment_momentum(num_days, sentiments):
    # Average of a given stock's sentiment over num_days. Each day is labeled
    # 1 if the sentiment that day is higher than the day before, and -1 if the
    # price is lower than the day before
    momentum = []
    sentiment_momentum = []
    for i in range(num_days, len(sentiments)):
        momentum.append(1 if sentiments[i] > sentiments[i - 1] else -1)

    for i in range(num_days, len(sentiments)):
        sentiment_momentum.append(mean(momentum[i - num_days:i]))

    return sentiment_momentum


def make_model(company, model, num_days, n, sentiment):

    df = pd.read_csv('../Consolidated_Data/' + company + '.csv')

    closing_prices = list(df['Close'])
    sentiments = list(df['Sentiment'])

    # Create feature vectors
    df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
    df = df[n:]
    df['Stock_Momentum'] = get_stock_momentum(n, closing_prices)
    df['Volatility'] = get_volatility(n, closing_prices)

    if sentiment:
        df = df[['Close', 'HL_PCT', 'PCT_change', 'Volatility', 'Stock_Momentum', 'Sentiment']]
        df['Sentiment_Momentum'] = get_sentiment_momentum(n, sentiments)
    else:
        df = df[['Close', 'HL_PCT', 'PCT_change', 'Volatility', 'Stock_Momentum']]

    df = df[:len(df)-num_days]

    # Convert input features into array and apply scaling
    X = np.array(df)
    X = preprocessing.scale(X)

    # Create Y vector; defined as whether a stock will increase or decrease in price in num_days
    Y = []
    for i in range(len(closing_prices)-num_days):
        Y.append(1 if closing_prices[i+num_days] > closing_prices[i] else -1)

    # Adjust length of Y to match X if needed
    if len(Y) > len(X):
        adjustment = len(Y) - len(X)
        Y = Y[adjustment:]

    # Split training and testing sets
    training_length = int(len(X) * 0.7)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

    X_train = np.array(X[0:training_length]).astype('float64')
    X_test = np.array(X[training_length:]).astype('float64')
    y_train = np.array(Y[0:training_length]).astype('float64')
    y_test = np.array(Y[training_length:]).astype('float64')

    # Construct and build classifier
    clf = svm.SVC(kernel='rbf', gamma='scale')
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    # print(len(y_test))
    return score


if __name__ == '__main__':
    company_list = ['Apple', 'Google', 'Amazon', 'Microsoft']
    sentiment_score = []
    baseline_score = []
    baseline_scores_dict = {'Company' : [],
                              'Forecast_Period': [],
                              'Num_Days': [],
                              'Mean': []}

    sentiment_scores_dict = {'Company' : [],
                              'Forecast_Period': [],
                              'Num_Days': [],
                              'Mean': []}

    forecast_ahead = [1, 5, 10, 20, 90, 180, 270]
    num_days_before = [1, 5, 10, 20, 90, 180, 270]

    for company in company_list:
        for forecast_period in forecast_ahead:
            for num_days in num_days_before:
                sentiment_score.append(make_model(company, "", forecast_period, num_days, True))
                baseline_score.append(make_model(company, "", forecast_period, num_days, False))

                sentiment_scores_dict['Company'].append(company)
                sentiment_scores_dict['Forecast_Period'].append(forecast_period)
                sentiment_scores_dict['Num_Days'].append(num_days)
                sentiment_scores_dict['Mean'].append(mean(sentiment_score))

                baseline_scores_dict['Company'].append(company)
                baseline_scores_dict['Forecast_Period'].append(forecast_period)
                baseline_scores_dict['Num_Days'].append(num_days)
                baseline_scores_dict['Mean'].append(mean(baseline_score))

                print("Baseline. Forecast Period = %d | Num Days Before = %d" % (forecast_period, num_days))
                print("Stats: mean %f median %f min %f max %f " % (mean(baseline_score), median(baseline_score),
                                                                 min(baseline_score), max(baseline_score)))
                print("Sentiment Model. Forecast Period = %d | Num Days Before = %d" % (forecast_period, num_days))
                print("Stats: mean %f median %f min %f max %f " % (mean(sentiment_score), median(sentiment_score),
                                                                 min(sentiment_score), max(sentiment_score)))
                sentiment_score = []
                baseline_score = []

    df_baseline = pd.DataFrame(baseline_scores_dict)
    df_sentiment = pd.DataFrame(sentiment_scores_dict)

    df_baseline.to_csv('../Results/Baseline_Scores.csv')
    df_sentiment.to_csv('../Results/Sentiment_Scores.csv')
    print("Done")
