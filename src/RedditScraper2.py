import requests
import pandas as pd
from datetime import datetime, timedelta

"""
Extracts top reddit posts related to Apple, Google, and Amazon stocks from 2010-2018.
Justin Huynh
"""


def get_posts(query="", subreddit="", before="", after="", size=""):
    resp = requests.get(
        "https://api.pushshift.io/reddit/search/submission/?q=" + query
        + "&subreddit=" + subreddit + "&before=" + before + "&after=" + after + "&sort_type=score" +
        "&size=" + size + "&sort=desc")
    return resp.json()


def get_apple():
    topics_dict = {"title": [],
                   "score": [],
                   "created": []}
    after = datetime.fromtimestamp(1262304000)   # 2010-01-01 00:00
    before = datetime.fromtimestamp(1262390400)  # 2010-01-02 00:00
    time_limit = datetime.fromtimestamp(1546300800)   # 2019-01-01 00:00

    while before < time_limit:
        posts = get_posts("Apple", "worldnews", str(int(before.timestamp())),
                          str(int(after.timestamp())), str(3))
        for submission in posts['data']:
            topics_dict['title'].append(submission['title'])
            topics_dict['score'].append(submission['score'])
            topics_dict['created'].append(datetime.strftime(datetime.fromtimestamp(submission['created_utc']),
                                                            '%Y-%m-%d'))
        print("Completed: ", after.timestamp())
        after += timedelta(days=1)
        before += timedelta(days=1)

    df = pd.DataFrame(topics_dict)
    df.to_csv("../Reddit_Posts/Worldnews_Apple.csv")


def get_google():
    topics_dict = {"title": [],
                   "score": [],
                   "created": []}
    after = datetime.fromtimestamp(1262304000)   # 2010-01-01 00:00
    before = datetime.fromtimestamp(1262390400)  # 2010-01-02 00:00
    time_limit = datetime.fromtimestamp(1546300800)   # 2019-01-01 00:00

    while before < time_limit:
        posts = get_posts("Google", "worldnews", str(int(before.timestamp())),
                          str(int(after.timestamp())), str(3))
        for submission in posts['data']:
            topics_dict['title'].append(submission['title'])
            topics_dict['score'].append(submission['score'])
            topics_dict['created'].append(datetime.strftime(datetime.fromtimestamp(submission['created_utc']),
                                                            '%Y-%m-%d'))
        print("Completed: ", after.timestamp())
        after += timedelta(days=1)
        before += timedelta(days=1)

    df = pd.DataFrame(topics_dict)
    df.to_csv("../Reddit_Posts/Worldnews_Google.csv")


def get_company_posts(company):
    topics_dict = {"title": [],
                   "score": [],
                   "created": []}
    after = datetime.fromtimestamp(1262304000)   # 2010-01-01 00:00
    before = datetime.fromtimestamp(1262390400)  # 2010-01-02 00:00
    time_limit = datetime.fromtimestamp(1546300800)   # 2019-01-01 00:00

    while before < time_limit:
        posts = get_posts(company, "worldnews", str(int(before.timestamp())),
                          str(int(after.timestamp())), str(3))
        for submission in posts['data']:
            topics_dict['title'].append(submission['title'])
            topics_dict['score'].append(submission['score'])
            topics_dict['created'].append(datetime.strftime(datetime.fromtimestamp(submission['created_utc']),
                                                            '%Y-%m-%d'))
        print("Completed: ", after.timestamp())
        after += timedelta(days=1)
        before += timedelta(days=1)

    df = pd.DataFrame(topics_dict)
    df.to_csv("../Reddit_Posts/Worldnews_" + company + ".csv")


def get_general_posts():
    """
    Retrieve general news headlines from Reddit Worldnews between
    2016-07-02 and 2018-12-31. Only get headlines from this
    date range because headlines from 2010-01-01 to 2016-06-01
    are already available in RedditNews.csv
    """
    topics_dict = {"title": [],
                   "score": [],
                   "created": []}
    after = datetime.fromtimestamp(1262304000)   # 2010-01-01 00:00
    before = datetime.fromtimestamp(1262390400)  # 2010-01-02 00:00
    time_limit = datetime.fromtimestamp(1546300800)   # 2019-01-01 00:00

    while before < time_limit:
        posts = get_posts(subreddit="worldnews", before=str(int(before.timestamp())),
                          after=str(int(after.timestamp())), size=str(10))
        for submission in posts['data']:
            topics_dict['title'].append(submission['title'])
            topics_dict['score'].append(submission['score'])
            topics_dict['created'].append(datetime.strftime(datetime.fromtimestamp(submission['created_utc']),
                                                            '%Y-%m-%d'))
        print("Completed: ", after.timestamp())
        after += timedelta(days=1)
        before += timedelta(days=1)

    df = pd.DataFrame(topics_dict)
    df.to_csv("../Reddit_Posts/Worldnews_All.csv")


def combine_general_news():
    df = pd.read_csv('../Reddit_Posts/RedditNews.csv')
    df = df[(df['Date'] >= '2010-01-01') & (df['Date'] <= '2016-06-01')]
    df.sort_values(by=['Date'], inplace=True, ascending=True)
    df.rename(index=str, columns={"Date":"created", "News":"title"}, inplace=True)
    df['score'] = 0

    df2 = pd.read_csv('../Reddit_Posts/Worldnews_All.csv')
    df2 = df2[['title', 'score', 'created']]
    df = df.append(df2)

    df.to_csv('../Reddit_Posts/Worldnews_All.csv')

if __name__ == '__main__':
    combine_general_news()
    # get_general_posts()
    # get_company_posts('Apple')
    # get_company_posts('Google')
    # get_company_posts('Amazon')
    # get_company_posts('Microsoft')

