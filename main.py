import pandas as pd
import numpy as np
import re, string, random
import cleantext

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from cleantext import clean

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

df_tweets.to_csv('clean_data.csv', encoding='utf-8', index=True)


# tokenize

# drop numbers?


def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


if __name__ == "__main__":

    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

    stop_words = stopwords.words('english')

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    freq_dist_pos = FreqDist(all_pos_words)
    print(freq_dist_pos.most_common(10))

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    train_data = dataset[:7000]
    test_data = dataset[7000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))
