#!/usr/bin/python3

import re
import os
import json

reObj1 = re.compile(r'[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)')
reObj2 = re.compile(r"(?:\n|[^\w\s])+")
processed = set() # lines which have been processed

def split_by_first_letter_in_username(inFile, outFile, letter):
    users = dict()
    for i, comment in enumerate(inFile):
        if i > 500:
            break
        if i in processed:
            continue
        userObj = json.loads(comment)
        username = userObj['author']
        if username.lower()[0] != letter:
            continue
        else:
            processed.add(i)
        userObj['body'] = re.sub(reObj1, ' ', userObj['body'])
        userObj['body'] = re.sub(reObj2, ' ', userObj['body'])
        if username in users:
            if userObj['subreddit'] not in users[username]['s']:
                users[username]['s'].append(userObj['subreddit'])
            users[username]['c'] += ' ' + userObj['body'].lower()
        else:
            users[username] = {
                's': [userObj['subreddit'].lower()],
                'c': userObj['body'].lower()
            }
    outFile.write(json.dumps(users, indent=4))


year = '2015'
fullpath = '/media/media/reddit-comments-dataset/reddit_data/{}/'.format(year)
with open(os.path.join(fullpath, 'RC_{}_trimmed'.format(year)), 'r') as inFile:
    for letter in 'abc':#'abcdefghijklmnopqrstuvwxyz':
        with open(os.path.join(os.path.join(fullpath, 'usernames_by_letter'), letter), 'w') as outFile:
            split_by_first_letter_in_username(inFile, outFile, letter)
        inFile.seek(0)
