# YouTube Clickbait Project
CS121 Spring 2019

### Python script

The clickbait_test_real.py script shows an exmaple of how the clickbait detection is done:

usage: clickbait_test_real --url URL

Provide the script with the youtube url to analyze. The script will print the model prediction: a number between 1 and 0, 1 is the video is 100% clickbait, 0 means the video is 0% clickbait.


Alternatively, you can use predict.py, if you don't want to install YouTube API related dependencies.

The predict.py script shows an example of how the clickbait detection is done:

usage: predict.py [-h] --title TITLE [--views VIEWS] [--likes LIKES]
                  [--dislikes DISLIKES] [--comments COMMENTS] [--imagepath IMAGEPATH]
                  
Provide the script with the title of the video to analyze and, if known, the number of views, likes, dislikes, comments and thumbnail image path. The script will print the model prediction: 1 if the video is probably clickbait, 0 otherwise.
