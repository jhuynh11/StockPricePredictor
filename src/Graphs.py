import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean


def compare_model_features():
    """
    Compares the performance of SVM models with/without the inclusion of the following features:
    - Index Momentum
    - Index Sentiment Momentum
    - Index Sentiment Volatility

    Results prepended with Y contain index momentum, index sentiment momentum, and index sentiment volatility
    :return:
    """

    df = pd.read_csv('../Results/Z_Baseline_scores.csv')
    df2 = pd.read_csv('../Results/Z_Sentiment_Scores.csv')

    df_list = list(df['Mean'])
    df_company = list(df['Company'])
    df_forecast = list(df['Forecast_Period'])
    df_num_days = list(df['Num_Days'])

    df2_list = list(df2['Mean'])
    df2_company = list(df2['Company'])
    df2_forecast = list(df2['Forecast_Period'])
    df2_num_days = list(df2['Num_Days'])

    s = []
    for i in range(len(df2_list)):
        if df2_list[i] > df_list[i]:
            s.append(i)

    res = []
    for i in range(len(s)):
        if df2_list[s[i]] - df_list[s[i]] > 0.1:
            res.append(df2_list[s[i]] - df_list[s[i]])
            print(df2_list[s[i]] - df_list[s[i]],
                  df2_company[s[i]], df2_forecast[s[i]],
                  df2_num_days[s[i]])
    print(min(res), max(res), mean(res))


compare_model_features()