# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 21:51:20 2018

@author: Paul
"""

from __future__ import division
import argparse
import pandas as pd

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
        
        self.nEmbed = nEmbed
        
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
        
        # defining a dictionary to match words with numbers
        self.dictionnary = {word: i for i, word in enumerate(self.word_count.index)}
        self.reverse_dictionnary = {i : word for i, word in enumerate(self.word_count.index)}
        
        # getting rid off unnecessary words in sentences
        self.clean_sentences = [[self.dictionnary[word] for word in sentence if word in self.dictionnary.keys()]\
                                for sentence in sentences]
        
        self.pairs = [(word1, word2) for sentence in self.clean_sentences for i1, word1 in enumerate(sentence) for i2, word2 in enumerate(sentence) if (np.abs(i1-i2) <= winSize) and i1!=i2]
        
        self.length_of_voc = len(self.dictionnary.keys())
        
        self.counting_matrix = np.zeros((self.length_of_voc, self.length_of_voc))
        
        for pair in self.pairs:
            self.counting_matrix[pair[0], pair[1]]+=1
            
        for i in range(self.length_of_voc):
            sum_i = sum(self.counting_matrix[:,i])
            self.counting_matrix[:,i]/=sum_i
       
        
        #raise NotImplementedError('implement it!')
    def softmax(self, x):
        decreased_x = x - np.max(x)
        return np.exp(decreased_x)/np.sum(np.exp(decreased_x))
    
    def mse(self, true_values, predictions):
        return np.mean(np.power(true_values - predictions, 2))

    def train(self,stepsize, epochs):
        self.input_weight = np.random.uniform(size = (self.length_of_voc, self.nEmbed))
        self.hidden_weight = np.random.uniform(size = (self.nEmbed, self.length_of_voc))
        # running through the epochs
        for epoch in range(epochs):
            # running through the training samples
            for word in range(self.length_of_voc):
                # creating the input vector
                input_vector = np.zeros((1, self.length_of_voc))
                input_vector[:,word] = 1
                # getting the output vector
                output_vector = self.counting_matrix[:,word].T
                # computing output
                first_step = np.dot(input_vector, self.input_weight)
                second_step = np.dot(first_step, self.hidden_weight)
                output_predictions = self.softmax(second_step)
                error = self.mse(output_vector, output_predictions)
        # raise NotImplementedError('implement it!')

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
model.word_count