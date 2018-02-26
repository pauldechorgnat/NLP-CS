# importing packages 
import pandas as pd 
import numpy as np
import nltk
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# loading the stopwords library
try :
    nltk.download('stopwords')

except :
    print("bah non en fait")

from nltk.corpus import stopwords

class Classifier:
    def __init__(self, layers = [20, 20, 10], size_of_voc = 300):
        self.first_layer = layers[0]
        self.layers = layers[1:]
    """Le Classifier"""
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        # Loading the training data
        train_data = pd.read_csv(trainfile, sep = "\t",
                                 names = ["sentiment", "subject", "timestamp", "word", "original_text"])
        
        # first lower the text 
        train_data['text'] = train_data['original_text'].apply(str.lower)
        # parse the words
        # we want to emphasize that there are special care to take about the word not and its contractions: 
        # it might be useful to keep them
        train_data['text'] = train_data["text"].apply(lambda sentence: sentence.replace("can\'t", "can not"))
        train_data['text'] = train_data["text"].apply(lambda sentence: sentence.replace("n\'t", " not"))
        train_data['words'] = train_data["text"].apply(lambda sentence:  "".join((char if char.isalpha() else " ") for char in sentence).lower().split() )
        
        # getting rid off stopwords
        self.stopwords = stopwords.words("english")
        self.stopwords.remove("not")
        
        train_data['words'] = train_data["words"].apply(lambda words : [word for word in words if word not in self.stopwords])
        
        # stemming the words with a Porter Stemmer
        stemmer = nltk.porter.PorterStemmer()
        train_data['stems'] = train_data["words"].apply(lambda words : [stemmer.stem(word) for word in words])
        
        # Changing the sentiment into a binary feature
        # 1 is for positive, 0 is for negative
        train_data['sentiment'] = (train_data['sentiment'] == "positive")*1
        
        # Storing the training data into an attribute of the Classifier
        self.data_train = train_data
        
        # keeping the categories that we will be trying to predict
        self.label_categories = pd.get_dummies(train_data['subject'])
        self.categories = self.label_categories.columns
        self.label_sentiment = train_data['sentiment']
        
        # perform wordcount
        # ça fait jamais de mal ...
        word_count = {}
        for row in train_data['stems']:
            for word in row:
                if word in word_count.keys():
                    word_count[word]+=1
                else:
                    word_count[word]=1
                
        self.vocabulary = np.unique(word_count.keys())
        self.word_count = word_count
        
        
    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        raise(NotImplemented('Implement it !'))




####### DEV MODE #####
classifier = Classifier()
classifier.train("../data/traindata.csv")

len(classifier.vocabulary)