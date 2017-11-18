#!/usr/bin/python3

import re
import os
import json

with open('hobbies.json', 'r') as hobbiesFile:
    hobbiesData = json.load(hobbiesFile)

hobbies = []
for key, value in hobbiesData.items():
    hobbies.extend(value)
hobbies = set(hobbies)

reObj1 = re.compile(r'[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)')
reObj2 = re.compile(r"(?:\n|[^\w\s'])+")
processed = set() # lines which have been processed

def split_by_first_letter_in_username(inFile, outFile, letter):
    users = dict()
    for i, comment in enumerate(inFile):
        if i == 5000000:
            break
        if i in processed:
            continue
        commentObj = json.loads(comment)
        username = commentObj['author']
        if username[0].lower() != letter and letter != '_':
            continue
        else:
            processed.add(i)
        if commentObj['subreddit'] in hobbies:
            if username in users:
                if commentObj['subreddit'].lower() not in users[username]['s']:
                    users[username]['s'].append(commentObj['subreddit'].lower())
            else:
                users[username] = {
                    's': [commentObj['subreddit'].lower()],
                    'c': ''
                }
        else:
            commentObj['body'] = re.sub(reObj1, ' ', commentObj['body'])
            commentObj['body'] = re.sub(reObj2, ' ', commentObj['body'])
            if username in users:
                users[username]['c'] += ' ' + commentObj['body'].lower()
            else:
                users[username] = {
                    's': [],
                    'c': commentObj['body'].lower()
                }
    filteredUsers = dict()
    for key, value in users.items():
        if len(value['s']) > 0 and len(value['c']) > 0:
            filteredUsers[key] = value
    print(len(filteredUsers.keys()))
    outFile.write(json.dumps(filteredUsers, indent=4))


year = '2015'
fullpath = '/media/media/reddit-comments-dataset/reddit_data/{}/'.format(year)
with open(os.path.join(fullpath, 'RC_{}_trimmed'.format(year)), 'r') as inFile:
    for letter in 'abcdefghijklmnopqrstuvwxyz_':
        with open(os.path.join(os.path.join(fullpath, 'usernames_by_letter'), letter), 'w') as outFile:
            split_by_first_letter_in_username(inFile, outFile, letter)
        inFile.seek(0)
