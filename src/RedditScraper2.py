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


if __name__ == '__main__':
    get_company_posts('Amazon')
    get_company_posts('Microsoft')
    # get_google()
    # get_apple()


