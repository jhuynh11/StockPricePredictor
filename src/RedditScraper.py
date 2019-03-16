import praw
import pandas as pd
from datetime import datetime

def subreddit_scrape(reddit, subreddit_name, limit):
    """
    Scrapes the subredits within subreddit_list
    :param reddit: Reddit object used to connect to the subreddit
    :param subreddit_name: Name of subreddit to scrape
    :param limit: Number of submissions to scrape
    :return: Pandas Dataframe of the scraped subreddit
    """

    subreddit = reddit.subreddit(subreddit_name)

    top_subreddit = subreddit.top(limit=limit,
                                  time_filter='all')

    # topics_dict = {"title": [],
    #                "score": [],
    #                "id": [],
    #                "url": [],
    #                "comms_num": [],
    #                "created": [],
    #                "body": [],
    #                "comments": []}

    topics_dict = {"title": [],
                   "score": [],
                   "created": []}

    for submission in top_subreddit:
        topics_dict["title"].append(submission.title)
        topics_dict["score"].append(submission.score)
        # topics_dict["id"].append(submission.id)
        # topics_dict["url"].append(submission.url)
        # topics_dict["comms_num"].append(submission.num_comments)
        topics_dict["created"].append(submission.created)
        # topics_dict["body"].append(submission.selftext)
        # topics_dict["comments"].append(submission.comments)

    return pd.DataFrame(topics_dict)

def subreddit_search(reddit, query, subreddit_name, limit):
    subreddit = reddit.subreddit(subreddit_name)

    topics_dict = {"title": [],
                   "score": [],
                   "created": []}

    for submission in subreddit.search(query=query, limit=limit):
        topics_dict["title"].append(submission.title)
        topics_dict["score"].append(submission.score)
        topics_dict["created"].append(submission.created)



if __name__ == '__main__':
    reddit = praw.Reddit(client_id = 'MOtOkwG9BPu5NQ',
                         client_secret = 'T5z_mQ87qSqlMM9kxDp3a7HWn-g',
                         user_agent = 'CSI4900')

    df = subreddit_scrape(reddit, 'Apple', limit=2000)
    df['created'] = df['created'].apply(lambda x: datetime.fromtimestamp(x))
    df.to_csv('../Reddit_Posts/Apple_Subreddit.csv')
    # print(df['title'])
    # for i in df['created']:
    #     print(datetime.utcfromtimestamp(i).strftime('%Y-%m-%d %H:%M:%S'))