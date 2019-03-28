# StockPricePredictor
CSI4900 Honours Research Project - University of Ottawa
This is a research project that explores the relationship between social media sentiment and stock price
prediction accuracy.

The goal of this study is to determine whether machine learning models such as Support Vector Machines (SVM) can
get higher classification accuracies with the inclusion of social media sentiment as an input feature. In other
words, can the public mood help predict whether a stock will increase or decrease in price in the future?

In summary, the study concludes that the addition of social media sentiment from Reddit posts can improve stock
price classification accuracies from 1 - 17%. On average, classification accuracies increase as the forecast period
increases as well. For example, predicting stock price movements 1 year into the future is more accurate than
predicting stock price movements 1 day into the future.


## Background
Recent research by Bollen et. al has discovered that Twitter sentiment analysis can help predict stock price movements 
for the Dow Jones Industrial Average (DJIA).  [Study Link](https://arxiv.org/abs/1010.3003)

*"Behavioral economics tells us that emotions can profoundly affect individual behavior and decision-making. Does this
also apply to societies at large, i.e., can societies experience mood states that affect their collective decision
making? By extension is the public mood correlated or even predictive of economic indicators?
Here we investigate whether measurements of collective mood states derived from large-scale Twitter feeds are
correlated to the value of the Dow Jones Industrial Average (DJIA) over time."*

We build upon the findings of this study by selecting a different social media platform for analysis: Reddit. 
Furthermore, we differentiate from Bollen's testing methodology by predicting stock price movements for specific
stocks, instead of the entire DJIA. 

## Datasets
We use two data types for our study: financial indicators and social media posts from Reddit.

### Financial Indicators
We take historical stock data from Yahoo Finance for 4 well-known technology stocks: Apple, Google, Amazon, and
Microsoft. This data contains High, Low, and Closing prices for these companies from 2010-2018. This data was extracted
by downloading the company's historical data in .csv format directly from the [Yahoo Finance website](https://ca.finance.yahoo.com/)


This data is held in the `Historical_Data` folder.


### Reddit Posts
Secondly, we take the top 3 Reddit posts from the Worldnews subreddit for each day from 2010-2018. These posts
are filtered by their content, and are chosen for their specific references to the 4 companies. For example, in getting
Apple specific posts, we filtered World News posts that only include content about Apple.

These Reddit posts were obtained by using the []Pushshift API to search through Reddit posts by date and content. The
source code for scraping these posts is in `src/RedditScraper2.py`.

This data is held in the `Reddit_Posts` folder.

## Methodology
First, we extract sentiment polarity from every single Reddit post by applying the VADER sentiment analysis library from
NLTK. This library analyzes the sentiment of a text string and outputs a sentiment polarity between -1 and 1, where
-1 is strongly negative sentiment and 1 is strongly positive sentiment. The intuition here is that stock prices should
rise when sentiment is strong and stock prices should fall when sentiment is negative. The code for performing 
sentiment analysis on each Reddit post is held in `src/Sentiment.py`


Second, we construct a classifier to predict stock price movements using Scikit-learn's Support Vector Machine library.
Historical stock data was preprocessed to extract key technical indicators such as momentum and volatility. The 
following features were used as input features to the baseline SVM model `src/SupportVectorMachine.py`:

1. Stock Momentum (Average of a stock's momentum  over the past n days)
2. Stock Volatility  (Average over the past n days of percent change in a stock's price per days)
3. High-Low Percent Change
4. Percent Change

We train the baseline model for the first 70% of data, and leave 30% for testing.

Third, we augment the baseline classifier using two sentiment related features:
1. Daily Sentiment
2. Sentiment Momentum (Average Sentiment for the past n days)

## Conclusions
The sentiment model can outperform the baseline model by 1-17%, depending on the forecast period. 
This demonstrates that the inclusion of Reddit sentiment for stock price machine learning models can
be a valuable input feature to improve classification accuracy.
