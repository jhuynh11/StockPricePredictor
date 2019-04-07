import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from statistics import mean, median

"""
SupportVectorMachine.py
Justin Huynh

Code for constructing support vector machine model using scikit-learn.
Contains code for calculating technical indicators such as momentum and volatility.
Main function initiates experimentation and parameter adjustments.
Output of experiment is stored in ../Results
"""

# DOW Jones Industrial Index (DJIA) data from 2010-2018. Used in many calculations so it is stored
# as a global variable here.
df_djia = pd.read_csv('../Consolidated_Data/DJIA.csv')


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


def get_sentiment_volatility(num_days, sentiments):
    volatility = []
    avg_volatility = []

    for i in range(num_days, len(sentiments)):
        if sentiments[i-1] != 0:
            volatility.append((sentiments[i] - sentiments[i-1])/sentiments[i-1])
        else:
            volatility.append(sentiments[i])

    for i in range(num_days, len(sentiments)):
        avg_volatility.append(mean(volatility[i - num_days:i]))

    return avg_volatility


def make_model(company, model, num_days, n, sentiment):

    df = pd.read_csv('../Consolidated_Data/' + company + '.csv')

    closing_prices = list(df['Close'])
    sentiments = list(df['Sentiment'])
    djia_sentiments = list(df_djia['Sentiment'])

    # Create feature vectors
    df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
    df = df[n:]
    df['Stock_Momentum'] = get_stock_momentum(n, closing_prices)
    df['Volatility'] = get_volatility(n, closing_prices)
    df['Index_Momentum'] = get_stock_momentum(n, df_djia['Close'])
    df['Index_Volatility'] = get_volatility(n, df_djia['Close'])

    if sentiment:
        # df = df[['Close', 'HL_PCT', 'PCT_change', 'Volatility', 'Stock_Momentum']]#, 'Sentiment']]
        df = df[['Volatility', 'Stock_Momentum', 'Index_Momentum', 'Index_Volatility']]
        df['Sentiment_Momentum'] = get_sentiment_momentum(n, sentiments)
        df['Sentiment_Volatility'] = get_sentiment_volatility(n, sentiments)
        df['Index_Sentiment_Momentum'] = get_sentiment_momentum(n, djia_sentiments)
        df['Index_Sentiment_Volatility'] = get_sentiment_volatility(n, djia_sentiments)
    else:
        df = df[['Volatility', 'Stock_Momentum', 'Index_Momentum', 'Index_Volatility']]

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

    # Calculate accuracy
    score = clf.score(X_test, y_test)
    return score


def run_experiment():
    company_list = ['Apple', 'Google', 'Amazon', 'Microsoft', 'DJIA']
    sentiment_score = []
    baseline_score = []

    max_baseline_score = []
    max_sentiment_score = []

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

    df_baseline.to_csv('../Results/Z_Baseline_Scores.csv')
    df_sentiment.to_csv('../Results/Z_Sentiment_Scores.csv')
    print("Done")


if __name__ == '__main__':
    run_experiment()
