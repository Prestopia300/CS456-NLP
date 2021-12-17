import os
import tweepy as tw
import pandas as pd
import time
import sys
import codecs
import re
import emoji
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer



# Clean the Text
def cleanTxtSymbols(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # Removed @mentions
    text = re.sub(r'#', '', text) # Removing the '#' symbol
    text = re.sub(r'RT[\s]+', '', text) # Removing RT
    text = re.sub(r'https?:\/\/\S+', '', text) # Remove the hyper link
    regrex_emoji_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags = re.UNICODE)
    text = regrex_emoji_pattern.sub(r'',text)
    text = emoji.get_emoji_regexp().sub(u'', text)
    
    return text

# I dont want to clean this too much, I want the regular sentence with caps and punct, 
# but remove all strange symbols, like : 
# repeated punct, 
def clean_tweet(tweet):
    # Clean each Sentence Seperately
    cleaned_sentences = []
    tweet_sentences = sent_tokenize(tweet)
    for sentence in tweet_sentences:
        #split into tokens by white space 
        tokens = sentence.split()
        # word conjunctions with '-' Ex: Super-duper
        tokens = [word.replace('-', ' ') for word in tokens]
        # prepare regex for char filtering 
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        # remove punctuation from each word 
        tokens = [re_punc.sub('', w) for w in tokens]
        # remove remaining tokens that are not alphabetic 
        tokens = [word for word in tokens if word.isalpha()]
        # # filter out stopwords 
        # stop_words = set(stopwords.words('english'))
        # tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens 
        tokens = [word for word in tokens if len(word) > 1]
        # Convert to Lower
        tokens = [w.lower() for w in tokens]

        # turn back into sentence (with period)
        clean_sent = ' '.join(tokens)
        clean_sent += '.'
        
        cleaned_sentences.append(clean_sent)

    # Put sentences together
    cleaned_paragraph = ''
    for sentence in cleaned_sentences:
        cleaned_paragraph += sentence + ' '
        
    return cleaned_paragraph
    

def write_to_pos(tweet, count):
    # Prepare path
    path = 'txt_sentoken/pos/' + "cv{}.txt".format(count)
    f = open(path, "w", encoding='utf-8')
    f.write(tweet)
    f.close()

def write_to_neg(tweet, count):
    # Prepare path
    path = 'txt_sentoken/neg/' + "cv{}.txt".format(count)
    f = open(path, "w", encoding='utf-8')
    f.write(tweet)
    f.close()

# ------------- TWEEPY -------------

# Enter authorizations
consumer_key="KnUlN9kEQ7gJdDiVp8oNE1Wg2"
consumer_secret="mRtOVvdiZIzqwdxebqo47WmOBGBvWjIwtZvufDU2E84vlm5QV4"
access_key="1027293645676769280-2heVzKV4eDN7LDv0Udtk8VkE8Ubj9m"
access_secret="6RdwPKQbgIX75EcsI3eovdsqhtomqUgsHWo4Lwa1lOwnM"

# Authenticate
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Define the search term and the date_since date as variables
search_words = 'taking a walk' + '-filter:retweets'
date_since = "2021-06-01"
NoOfTerms = 2800

# Collect tweets
tweets = tw.Cursor(api.search, tweet_mode='extended', q=search_words, lang="en", since=date_since).items(NoOfTerms)

tweet_array = [tweet.full_text for tweet in tweets]

# Iterate and Clean and Print tweets
all_tweets= []
for tweet in tweet_array:
    tweet_2 = cleanTxtSymbols(tweet)
    tweet_2 = clean_tweet(tweet_2)
    print(tweet_2)
    print()
    all_tweets.append(tweet_2)



# Delete all previously stored files
pos_path = 'txt_sentoken/pos'
for f in os.listdir(pos_path):
    os.remove(os.path.join(pos_path, f))
neg_path = 'txt_sentoken/neg'
for f in os.listdir(neg_path):
    os.remove(os.path.join(neg_path, f))


pos_count = 0
neg_count = 0
for tweet in all_tweets:
    # Use VADER to calculate sentiment of tweet
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(tweet)
    comp = ss['compound']

    # if neutral tweet
    if comp > -0.1 and comp < 0.1:
        continue

    # if positive tweet
    if comp > 0:
        # add to pos
        write_to_pos(tweet, pos_count)
        pos_count += 1

    # of negative tweet
    if comp <= 0:
        # add to neg
        write_to_neg(tweet, neg_count)
        neg_count += 1



# Garbage Bin Code : 

# Collect a list of tweets
# all_tweets = [(tweet.text).encode("utf-8") for tweet in tweets]