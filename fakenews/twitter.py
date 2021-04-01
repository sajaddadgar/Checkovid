from .config import *
import requests
import json


def get_bearer_token():
    # return "AAAAAAAAAAAAAAAAAAAAAKFWHQEAAAAAv7FeKFJ7Abblj%2F%2FQYgxv6M5lc2o%3D59rCfbWhZy60mA6wjv3ofmpi2Z6wcFefjppgUT9JVmVwlPTr3H"
    response = requests.post(
        "https://api.twitter.com/oauth2/token",
        auth=(consumer_key, consumer_secret),
        data={'grant_type': 'client_credentials'},
        headers={"User-Agent": "TwitterDevCovid19StreamQuickStartPython"})

    if response.status_code is not 200:
        print("Cannot get a Bearer token (HTTP {}): {}".format(response.status_code, response.text))

    body = response.json()
    return body['access_token']


def get_tweets_by_id(id):
    response = requests.get(
        'https://api.twitter.com/1.1/statuses/show.json?id=' + id + '&tweet_mode=extended',
        headers={"Authorization": "Bearer {}".format(
            get_bearer_token())}, stream=True)
    tweets_list = []
    for response_line in response.iter_lines():
        if response_line:
            tweets = json.loads(response_line)
            tweets_list.append(tweets)
            return tweets


def get_user_by_id(id):
    response = requests.get(
        'https://api.twitter.com/1.1/users/lookup.json?user_id=' + id,
        headers={"Authorization": "Bearer {}".format(
            get_bearer_token())}, stream=True)

    for response_line in response.iter_lines():
        if response_line:
            user = json.loads(response_line)
            return user[0]
