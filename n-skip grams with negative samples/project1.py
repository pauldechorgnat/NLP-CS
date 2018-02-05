# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 21:51:20 2018

@author: Paul

https://arxiv.org/pdf/1411.2738.pdf
"""

from __future__ import division
import argparse
import pandas as pd

import datetime

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize


__authors__ = ['paul_dechorgnat','victor_terras-dober','dimitri_trotignon']
__emails__  = ['fatherchristmoas@northpole.dk','toothfairy@blackforest.no','easterbunny@greenfield.de']

def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append( "".join((char if char.isalpha() else " ") for char in l).lower().split() )
    return sentences

def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs


class mySkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 15):
        
        # negative rate is the number of negative examples we need to create for one positive example
        # nEmbed is the number of neurons within the embedding layer
        # winSize is the size of the context
        # minCount is the minimum number of instances required for a word to be kept in the training data
        self.negativeRate = negativeRate
        self.nEmbed = nEmbed
        self.winSize = winSize
        self.minCount = minCount
        
        self.C = 2*winSize # length of context
        
         # defining a list of stopwords to get rid off
        stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
             'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
             'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
             'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
             'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
             'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
             'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
             'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
             'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
             'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
             'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'a']
        
        
        # computing wordcount and size of the vocabulary        
        self.word_count = {}
        
        for sentence in sentences:
            for word in sentence:
                if word in self.word_count.keys():
                    self.word_count[word] += 1
                else:
                    self.word_count[word] = 1
        
        self.word_count = pd.Series(self.word_count)
        
        
        
        # getting rid of too rare words
        self.word_count = self.word_count[self.word_count >= minCount]
        # keeping only valid words
        valid_words = [word for word in self.word_count.index if word not in stopwords]
        self.word_count = self.word_count.loc[valid_words].sort_values(ascending = False)
                
        # keeping the length of the vocabulary
        self.V = len(valid_words)
        
        # defining a dictionary to match words with numbers
        self.dictionnary = {word: i for i, word in enumerate(self.word_count.index)}
        self.reverse_dictionnary = {i : word for i, word in enumerate(self.word_count.index)}

        # getting rid off unnecessary words in sentences
        self.clean_sentences = [[self.dictionnary[word] for word in sentence if word in self.dictionnary.keys()]\
                                for sentence in sentences]
        
        self.contexts = []
        for sentence in self.clean_sentences:
            for i1, target_word in enumerate(sentence):
                self.contexts.append((target_word, [context_word for i2, context_word in enumerate(sentence) if (np.abs(i1-i2)<=winSize) and i1 != i2]))
        
#        http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
#        self.contingence_matrix = np.zeros((self.V, self.V))
#        
#        for target_word, context in self.contexts:
#            for context_word in context :
#                self.contingence_matrix[target_word, context_word] += 1

        # we want to have a probability of creating negative samplings 
        self.negative_probability = np.power(self.word_count, 3/4)
        self.negative_probability /= np.sum(self.negative_probability)
        self.negative_probability.index = [self.dictionnary[word] for word in self.negative_probability.index]
        
        # initializing weights 
        self.input2hidden_weights = np.random.uniform(size = (self.V, self.nEmbed))
        self.hidden2output_weights = np.random.uniform(size = (self.nEmbed, self.V))
       
    def create_negative_samples(self, context_word):
        # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/        
        negative_samples = np.random.choice(a = self.negative_probability.index,
                                            p = self.negative_probability,
                                            size =  self.negativeRate)
        return [(context_word, 1)] + [(negative_word, 0) for negative_word in negative_samples]
        
    def sigmoid(self, x, y):
        return 1/(1+np.exp(-np.sum(x*y.T)))
    def train(self,stepsize, epochs):
        
        # running through the epochs
        for epoch in range(epochs):
            if epoch % 10 == 0:
                print("Epoch n° : {}/{} - {}".format(epoch, epochs, str(datetime.datetime.now())))
            
            # running through contexts
            for target_word, context in self.contexts:
                
                h = self.input2hidden_weights[target_word,:]
                # creating negative samples 
                # !!!! there is something fuzzy : the world can be a negative and a positive sample here
                for context_word in context:
                    # generating negative samples
                    training_outputs = self.create_negative_samples(context_word)
                    # computing EH
                    EH = np.sum([(self.sigmoid(self.hidden2output_weights[:,j], h) - tj)*self.hidden2output_weights[:,j] for j, tj in training_outputs], axis = 0)
                    
                    # updating output layer weights 
                    for j, tj in training_outputs:
                        self.hidden2output_weights[:,j] -= stepsize * (self.sigmoid(self.hidden2output_weights[:,j], h)-tj) * h.T
                    
                    # updating input layer wiegths
                    self.input2hidden_weights[target_word, :] -= stepsize * EH.T
                

    def save(self,path):
        raise NotImplementedError('implement it!')

    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        raise NotImplementedError('implement it!')

    @staticmethod
    def load(path):
        raise NotImplementedError('implement it!')

#if __name__ == '__main__':
#
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--text', help='path containing training data', required=True)
#    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
#    parser.add_argument('--test', help='enters test mode', action='store_true')
#
#    opts = parser.parse_args()
#
#    if not opts.test:
#        sentences = text2sentences(opts.text)
#        sg = mySkipGram(sentences)
#        sg.train()
#        sg.save(opts.model)
#
#    else:
#        pairs = loadPairs(opts.text)
#
#        sg = mSkipGram.load(opts.model)
#        for a,b,_ in pairs:
#            print(sg.similarity(a,b))


path = "C:/Users/Paul/Desktop/MSc DSBA/10. Natural Language Processing/Github/NLP-CS/n-skip grams with negative samples" + "/total_data.txt"

sentences = text2sentences(path)

model = mySkipGram(sentences)
print("J'ai fini pour le moment")