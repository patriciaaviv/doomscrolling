import pandas as pd
import numpy as np
import re, string, random
import cleantext
import nltk

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk import FreqDist, classify, NaiveBayesClassifier
from cleantext import clean
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

# Read data from yesterday
tweets_2204 = pd.read_csv('0422_UkraineCombinedTweetsDeduped.csv', low_memory=False)

# drop unnecessary columns
df_tweets = tweets_2204.drop(
    ['userid', 'acctdesc', 'location', 'usercreatedts', 'tweetid', 'tweetcreatedts', 'coordinates', 'extractedts',
     'is_retweet',
     'original_tweet_id', 'original_tweet_userid', 'original_tweet_username',
     'in_reply_to_status_id', 'in_reply_to_user_id',
     'in_reply_to_screen_name', 'is_quote_status'], axis=1)

# drop non-english tweets
df_tweets = df_tweets[df_tweets['language'] == "en"]

# TODO: for later, check if user is bot, for now skipping

# drop non-ascii characters, if ascii character is detected -> replace with ""
df_tweets['text'] = df_tweets['text'].replace(to_replace="/[^ -~]+/g", value="", regex=True)

# remove emojis
filter_char = lambda c: ord(c) < 256
df_tweets['text'] = df_tweets['text'].apply(lambda s: ''.join(filter(filter_char, s)))
df_tweets['text'].astype(str).apply(lambda x: x.encode('latin-1', 'ignore').decode('latin-1'))

# drop tweets that are now empty
df_tweets = df_tweets[df_tweets['text'] != ""]

# remove mentions, underscore from mentions remains?
df_tweets['text'] = df_tweets['text'].str.replace('@[A-Za-z0-9]+\s?', '', regex=True)

# remove underscore
df_tweets['text'] = df_tweets['text'].str.replace('_', '', regex=True)

# remove hashtags
df_tweets['text'] = df_tweets['text'].str.replace('#', '', regex=True)

df_tweets['text'] = df_tweets['text'].str.replace('.', '', regex=True)
df_tweets['text'] = df_tweets['text'].str.replace(',', '', regex=True)

# remove html code
# df_tweets['text'] = [re.sub(r'/&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-fA-F]{1,6});/ig','', str(x)) for x in df_tweets['text']]
# df_tweets['text'].replace({r"/&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-fA-F]{1,6});/ig": ''}, inplace=True, regex=True)
# both approaches don't work, fix with hardcode for now
df_tweets['text'] = df_tweets['text'].str.replace('&amp', '', regex=True)
df_tweets['text'] = df_tweets['text'].str.replace('&gt', '', regex=True)
df_tweets['text'] = df_tweets['text'].str.replace('&lt', '', regex=True)

# letters to lower case
df_tweets['text'] = df_tweets['text'].astype(str).str.lower()

# remove stop words
# throws warning: FutureWarning: The default value of regex will change from True to False in a future version, disregard
stop = stopwords.words('english')
pat_stopwords = r'\b(?:{})\b'.format('|'.join(stop))
df_tweets['text'] = df_tweets['text'].str.replace(pat_stopwords, '')
df_tweets['text'] = df_tweets['text'].str.replace(r'\s+', ' ')

# tokenize our tweets data frame
regexp = RegexpTokenizer('\w+')
df_tweets_tokenized = df_tweets['text'].apply(regexp.tokenize)

# remove infrequent words, doesn't work rn
# df_tweets_tokenized['text'] = df_tweets_tokenized['text'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))

# TODO: drop numbers?

# make new csv with clean data set for testing
df_tweets.to_csv('clean_data.csv', encoding='utf-8', index=True)
df_tweets_tokenized.to_csv('clean_data_tokenize.csv', encoding='utf-8', index=True)
print('finished csv transformation')

# more accuracy -> implement lemmatization




