#!/usr/bin/python3

import json

year = '2015'
fullpath = '/media/media/reddit-comments-dataset/reddit_data/{}/RC_{}'.format(year, year)
with open(fullpath, 'r') as inFile, open(fullpath + '_trimmed', 'w') as outFile:
    for i, comment in enumerate(inFile):
        newUserObj = dict()
        oldUserObj = json.loads(comment)
        for key, value in oldUserObj.items():
            if key in {'author', 'subreddit', 'body'}:
                newUserObj[key] = value
        outFile.write(json.dumps(newUserObj) + '\n')
