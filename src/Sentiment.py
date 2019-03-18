import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def classify_apple():
    """
    Classifies the sentiment of every post title in a given file.
    Output: Exports the sentiment file into Reddit_Posts
    """
    df = pd.read_csv("../Reddit_Posts/Test.csv")
    # Add a sentiment column
    df['Sentiment'] = 0.0
    sid = SentimentIntensityAnalyzer()

    for index, row in df.iterrows():
        score = sid.polarity_scores(row['title'])['compound']
        df.at[index, 'Sentiment'] = score
    df.to_csv("../Reddit_Posts/Apple_Sentiment.csv")


def consolidate_apple():
    # Aggregate sentiment of posts by post date
    sentiment_df = pd.read_csv("../Reddit_Posts/Apple_Sentiment.csv")
    sentiment_df = sentiment_df.groupby(['created']).mean()

    # Merge sentiment data with stock data
    stock_df = pd.read_csv("../Historical_Data/AAPL.csv")
    consolidated = stock_df.merge(sentiment_df, how='left', left_on='Date', right_on='created')
    # Drop unneeded columns
    consolidated.drop(['Unnamed: 0', 'Unnamed: 0.1', 'score'], axis=1, inplace=True)
    # Replace null sentiments with 0
    consolidated['Sentiment'].fillna(0, inplace=True)
    consolidated.to_csv("../Consolidated_Data/Apple.csv")


classify_apple()
consolidate_apple()

# classify_apple()

