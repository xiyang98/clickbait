import csv
import requests
import os
import pickle
import numpy as np
import pandas as pd
import re
import emoji
from gensim.parsing.preprocessing import *
from sklearn.model_selection import train_test_split
# if not os.path.exists('non_clickbait'):
#     os.makedirs('non_clickbait')
# if not os.path.exists('clickbait'):
#     os.makedirs('clickbait')

# with open('non_clickbait.csv', newline='', encoding = "ISO-8859-1") as csvfile:
#     clickbaitreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
#     for row in clickbaitreader:
#         video_id = row[7]
#         if "=" == video_id[0]:
#             video_id = video_id[1:]
#         image_url = row[11]
#         img_data = requests.get(image_url).content
#         with open('non_clickbait/'+video_id+'.jpg', 'wb') as handler:
#             handler.write(img_data)

# with open('clickbaits.csv', newline='') as csvfile:
#     clickbaitreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
#     for row in clickbaitreader:
#         video_id = row[7]
#         image_url = row[11]
#         img_data = requests.get(image_url).content
#         with open('clickbait/'+video_id+'.jpg', 'wb') as handler:
#             handler.write(img_data)





# if not os.path.exists('train'):
#     os.makedirs('train')
# if not os.path.exists('train/clickbait'):
#     os.makedirs('train/clickbait')
# if not os.path.exists('train/non_clickbait'):
#     os.makedirs('train/non_clickbait')
# if not os.path.exists('test'):
#     os.makedirs('test')
# if not os.path.exists('test/clickbait'):
#     os.makedirs('test/clickbait')
# if not os.path.exists('test/non_clickbait'):
#     os.makedirs('test/non_clickbait')
# if not os.path.exists('val'):
#     os.makedirs('val') 
# if not os.path.exists('val/clickbait'):
#     os.makedirs('val/clickbait')
# if not os.path.exists('val/non_clickbait'):
#     os.makedirs('val/non_clickbait')


# non_clickbait.csv

def tokenize(string):

    """ Tokenizes a string.

    Adds a space between numbers and letters, removes punctuation, repeated whitespaces, words shorter than 2
    characters, and stop-words. Returns a list of stems and, eventually, emojis.

    @param string: String to tokenize.
    @return: A list of stems and emojis.
    """

    # Based on the Ranks NL (Google) stopwords list, but "how" and "will" are not stripped, and words shorter than 2
    # characters are not checked (since they are stripped):
    stop_words = [
        "about", "an", "are", "as", "at", "be", "by", "com", "for", "from", "in", "is", "it", "of", "on", "or", "that",
        "the", "this", "to", "was", "what", "when", "where", "who", "with", "the", "www"
    ]

    string = strip_short(
        strip_multiple_whitespaces(
            strip_punctuation(
                split_alphanum(string))),
        minsize=2)
    # Parse emojis:
    emojis = [ c for c in string if c in emoji.UNICODE_EMOJI ]
    # Remove every non-word character and stem each word:
    string = stem_text(re.sub(r"[^\w\s,]", "", string))
    # List of stems and emojis:
    tokens = string.split() + emojis
    
    for stop_word in stop_words:
        try:
            tokens.remove(stop_word)
        except:
            pass

    return tokens


non_clickbait_df = pd.read_csv('non_clickbait.csv', names = ['channel_id','channel_name','channel_subscribers', 
                                                      'channel_videos','channel_views','video_comments',
                                                      'video_dislikes','video_id','video_likes','video_title',
                                                      'video_views', 'image_url'], encoding = "ISO-8859-1")



clickbait_df = pd.read_csv('clickbaits.csv', names = ['channel_id','channel_name','channel_subscribers', 
                                                      'channel_videos','channel_views','video_comments',
                                                      'video_dislikes','video_id','video_likes','video_title',
                                                      'video_views', 'image_url'], encoding = "ISO-8859-1")


# clickbait metadata

non_clickbait_df.drop(columns=['image_url'], inplace=True)
non_clickbait_df["video_title_tokenized"] = non_clickbait_df["video_title"].apply(tokenize)


clickbait_df.drop(columns=['image_url'], inplace=True)
clickbait_df["video_title_tokenized"] = clickbait_df["video_title"].apply(tokenize)

clickbait_df["label"] = 1
non_clickbait_df["label"] = 0

clickbait_df = clickbait_df.sample(frac=1)
non_clickbait_df = non_clickbait_df.sample(frac=1)

dataframe = pd.concat([clickbait_df, non_clickbait_df]).sample(frac=1).sample(frac=1)

X_train, X_test, y_train, y_test = train_test_split(
    dataframe.loc[:, dataframe.columns != "label"], 
    dataframe["label"], 
    test_size=0.2, 
    random_state=99)

pickle.dump(X_train, open("x-train", "wb"))
pickle.dump(y_train, open("y-train", "wb"))
pickle.dump(X_test, open("x-test", "wb"))
pickle.dump(y_test, open("y-test", "wb"))


