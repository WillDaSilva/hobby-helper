#!/usr/bin/python3

import os
import json
import numpy
from random import shuffle
import itertools as it

from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

with open('hobbiesReversed.json', 'r') as reversedHobbiesFile:
    reversed_hobbies = json.load(reversedHobbiesFile)

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

    def to_array(self):
        self.sentences = []
        self.labels = {}
        for source, prefix in self.sources.items():
            with open(source, 'r') as inFile:
                inData = json.load(inFile)
            for i, userdata in enumerate(inData.values()):
                self.sentences.append(LabeledSentence(utils.to_unicode(userdata['c']).split(), [prefix + '_%s' % i]))
                self.labels[prefix + '_%s' % i] = it.chain.from_iterable([reversed_hobbies[s] for s in userdata['s']])
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

dirpath = '/media/media/reddit-comments-dataset/reddit_data/2015/test2_usernames_by_letter/'
sources = {dirpath + x : '2015-' + x for x in 'abcdefghijklmnopqrstuvwxyz_'}

sentences = LabeledLineSentence(sources)

model = Doc2Vec(min_count = 3, window = 10, size = 200, sample = 1e-2, negative=10, workers = 8)

model.build_vocab(sentences.to_array())

print('Starting training')
model.train(sentences.sentences_perm(),total_examples=model.corpus_count,epochs=20)
print('Done training. Saving model')
model.save_word2vec_format('redditmodel.d2v', doctag_vec=True, binary=True)

labelArray = []
vectorArray = []

for key in labels.keys():
	labelArray.append(labels[key]);
	vectorArray.append(model.docvecs[key])

with open('vectorsAndLabels.json', 'w') as vl:
	vl.write(json.dumps(vectorArray))
	vl.write(json.dumps(labelArray))