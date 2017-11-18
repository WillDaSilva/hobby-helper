# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

import numpy
import json

from random import shuffle

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        '''
        flipped = {}
        
        # make sure that keys are 
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
                '''
    '''
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    '''
    
    def to_array(self):
        self.sentences = []
        self.labels = {}
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    userObj = json.loads(line)
                    self.sentences.append(LabeledSentence(utils.to_unicode(userObj.c).split(), [prefix + '_%s' % item_no]))
                    self.labels[prefix + '_%s' % item_no] = userObj.s
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

sources = {for x : '2015-' + x in 'abcdefghijklmnopqrstuvwxyz_'}

sentences = LabeledLineSentence(sources)

model = Doc2Vec(min_count = 3, window = 10, size = 200, sample = 1e-2, negative=10, workers = 8)

model.build_vocab(sentences.to_array())


model.train(sentences.sentences_perm(),total_examples=model.corpus_count,epochs=20)

model.save('./redditmodel.d2v')
