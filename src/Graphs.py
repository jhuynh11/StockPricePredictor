import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean, median


def compare_model_features():
    """
    Compares the performance of SVM models with/without the inclusion of the following features:
    - Index Momentum
    - Index Sentiment Momentum
    - Index Sentiment Volatility

    Results prepended with Y contain index momentum, index sentiment momentum, and index sentiment volatility
    :return:
    """

    df = pd.read_csv('../Results/Z_Baseline_Scores.csv')
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
        if df2_list[s[i]] - df_list[s[i]] > 0:
            res.append(df2_list[s[i]] - df_list[s[i]])
            print(df2_list[s[i]] - df_list[s[i]],
                  df2_company[s[i]], df2_forecast[s[i]],
                  df2_num_days[s[i]])
    print(min(res), max(res), mean(res))
    print(len(res), len(df2_list))


def visualize():
    df = pd.read_csv('../Results/DJIA.csv')
    df['Close'].plot()
    plt.show()


def refresh(dictionary):
    for key in dictionary:
        dictionary[key] = []
    return dictionary


def get_key_metrics():
    df = pd.read_csv('../Results/Z_Baseline_scores.csv')
    df2 = pd.read_csv('../Results/Z_Sentiment_Scores.csv')

    scores = [('Baseline', df), ('Sentiment', df2)]
    forecast_periods = [1, 5, 10, 20, 90, 180, 270]
    num_days = [1, 5, 10, 20, 90, 180, 270]

    results = {'Forecast_Period' : [],
               'Num_Days': [],
               'Mean': [],
               'Median': [],
               'Min': [],
               'Max': []}

    for score in scores:
        for fp in forecast_periods:
            for n in num_days:
                forecast_filter = score[1]['Forecast_Period'] == fp
                days_filter = score[1]['Num_Days'] == n
                frame = score[1][forecast_filter & days_filter]
                results['Forecast_Period'].append(fp)
                results['Num_Days'].append(n)
                results['Mean'].append(mean(frame['Mean']))
                results['Median'].append(median(frame['Mean']))
                results['Min'].append(min(frame['Mean']))
                results['Max'].append(max(frame['Mean']))
        df_out = pd.DataFrame(results)
        df_out.to_csv('../Results/' + score[0] + '_Summary.csv')
        results = refresh(results)

    print('Exported key summary stats')

# visualize()
# compare_model_features()
get_key_metrics()