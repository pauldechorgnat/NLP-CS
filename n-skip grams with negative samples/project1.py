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

### Importing a new package for visualization => to remove later...
from tqdm import tqdm as prog_bar

### we are going to use the json library to save the model 
import json


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
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize = 2, minCount = 15):
        
        # negative rate is the number of negative examples we need to create for one positive example
        # nEmbed is the number of neurons within the embedding layer
        # winSize is the size of the context
        # minCount is the minimum number of instances required for a word to be kept in the training data
        self.negativeRate = negativeRate
        self.nEmbed = nEmbed
        self.winSize = winSize
        self.minCount = minCount
        self.subSamplingThreshold = .00001
        
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
        print("Starting word count")
        self.word_count = {}        
        for sentence in prog_bar(sentences):
            for word in sentence:
                if word in self.word_count.keys():
                    self.word_count[word] += 1
                else:
                    self.word_count[word] = 1
        
        self.word_count = pd.Series(self.word_count)
        print("Word count done")
        
        print("Starting word cleaning")
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
        print("Word cleaning done")
        
        print("Starting sub-sampling")
        # performing sub sampling
        self.frequencies = self.word_count / np.sum(self.word_count)
        self.clean_sentences = [[word for word in sentence if not self.remove_word_subsampling(word)] for sentence in self.clean_sentences]
        print("Sub-sampling done")
        
        print("Starting contexts computation")
        self.contexts = []
        for sentence in prog_bar(self.clean_sentences):
            for i1, target_word in enumerate(sentence):
                self.contexts.append((target_word, [context_word for i2, context_word in enumerate(sentence) if (np.abs(i1-i2)<=self.C/2) and i1 != i2]))
        print("Contexts computation done")
        
        print("Starting probability computation for negative sampling")
        # we want to have a probability of creating negative samplings 
        self.negative_probability = np.power(self.word_count, 3/4)
        self.negative_probability /= np.sum(self.negative_probability)
        self.negative_probability.index = [self.dictionnary[word] for word in self.negative_probability.index]
        print("Probability Computation for negative sampling done")
        
        print("Starting weights initialization")
        # initializing weights 
        self.input2hidden_weights = np.random.uniform(size = (self.V, self.nEmbed))
        self.hidden2output_weights = np.random.uniform(size = (self.nEmbed, self.V))
        print("Weights initialization done")
       
    def create_negative_samples(self, context_word):
        # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/        
        negative_samples = np.random.choice(a = self.negative_probability.index,
                                            p = self.negative_probability,
                                            size =  self.negativeRate)
        return [(context_word, 1)] + [(negative_word, 0) for negative_word in negative_samples]
        
    def sigmoid(self, x, y):
        return 1/(1+np.exp(-np.sum(x*y.T)))
    
    def remove_word_subsampling(self, word):
        frequency = self.frequencies.iloc[word]
        if frequency > self.subSamplingThreshold:
            probability = (frequency - self.subSamplingThreshold)/frequency - np.sqrt(self.subSamplingThreshold/frequency)
            return np.random.uniform()<probability
        else:
            return False
    
    
    def train(self,stepsize, epochs):
        
        # running through the epochs
        for epoch in range(epochs):
            if epoch % 10 == 0:
                print("Epoch n° : {}/{} - {}".format(epoch+1, epochs, str(datetime.datetime.now())))
            
            # running through contexts
            for target_word, context in prog_bar(self.contexts):
                
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
                
        print("Training ended at ",str(datetime.datetime.now()))
        pass
    
    def save(self,path):
        # keeping all the attributes of the model
        parameters = self.__dict__.copy()
        
        # changing some types in order to get them into a json 
        parameters['word_count'] = parameters['word_count'].to_json()
        parameters['frequencies'] = parameters['frequencies'].to_json()
        parameters['negative_probability'] = parameters['negative_probability'].to_json()
        parameters['input2hidden_weights'] = parameters['input2hidden_weights'].tolist()
        parameters['hidden2output_weights'] = parameters['hidden2output_weights'].tolist()
        
        # writing parameters
        with open(path, 'w') as file:
            file.write(json.dumps(parameters))
        file.close()
        print("Model saved")
        pass
        
    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        # for this function we are going to compute the cosine distance between the two words vector
        if (word1 not in self.dictionnary.keys()):
            print("'{}' is not in the original corpus.".format(word1))
            pass
        if (word2 not in self.dictionnary.keys()):
            print("'{}' is not in the original corpus.".format(word2))
            pass
        
        word1_vector = self.input2hidden_weights[self.dictionnary[word1],:]
        word2_vector = self.input2hidden_weights[self.dictionnary[word2],:]
        cosine_distance = np.dot(word1_vector, word2_vector)/ np.linalg.norm(word1_vector)/np.linalg.norm(word2_vector)
        
        # cosine distance is between -1 and 1 so we need to 
        return (1+cosine_distance)/2
    
    @staticmethod
    def load(path):
        # reading parameters
        with open(path, 'r') as file:
            parameters = json.load(file)
        file.close
        # instantiating without calling __init__ constructor
        new_skip_gram = mySkipGram.__new__(mySkipGram)
        
        # changing some parameters into their normal type
        parameters['word_count'] = pd.read_json(parameters['word_count'], typ= 'series', orient = 'records')
        parameters['frequencies'] = pd.read_json(parameters['frequencies'], typ= 'series', orient = 'records')
        parameters['negative_probability'] = pd.read_json(parameters['negative_probability'], typ= 'series', orient = 'records')
        parameters['input2hidden_weights'] = np.array(parameters['input2hidden_weights']).reshape(parameters["V"], parameters["nEmbed"])
        parameters['hidden2output_weights'] = np.array(parameters['hidden2output_weights']).reshape(parameters["nEmbed"], parameters["V"])
        
        # setting attributes to the new instance
        for attribute, attribute_value in parameters.items():
            setattr(new_skip_gram, attribute, attribute_value)
        print("Model loaded")
        return new_skip_gram


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


path = "C:/Users/Paul/Desktop/MSc DSBA/10. Natural Language Processing/Github/NLP-CS/n-skip grams with negative samples/total_data.txt"

sentences = text2sentences(path)

model = mySkipGram(sentences)
model.save("C:/Users/Paul/Desktop/MSc DSBA/10. Natural Language Processing/Github/NLP-CS/n-skip grams with negative samples/first_model.txt")

model2 = mySkipGram.load("C:/Users/Paul/Desktop/MSc DSBA/10. Natural Language Processing/Github/NLP-CS/n-skip grams with negative samples/first_model.txt")
print("J'ai fini pour le moment")