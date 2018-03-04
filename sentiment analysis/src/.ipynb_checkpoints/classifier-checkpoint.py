# importing packages 
import pandas as pd 
import numpy as np
import nltk
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import spacy
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# loading the stopwords library
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

print("Loading Spacy Model")
nlp = spacy.load('en_core_web_lg')
print("Spacy model loaded")

# defining a function to get the word embedding of the words
def get_word_embeddings(stems):
    vectors = []
    for stem in stems: 
        token = nlp(stem)
        vectors.append(token.vector)
    return vectors


# function to format the data 
def formatting_data(path):
    print("Loading data ...")
    data = pd.read_csv(path, sep = "\t",
                       names = ["sentiment", "subject", "word", "timestamp", "original_text"])
    print("Data loaded")

    # first lower the text 
    print("Text tokenization ...")
    data['text'] = data['original_text'].apply(str.lower)
    # parse the words
    # we want to emphasize that there are special care to take about the word not and its contractions: 
    # it might be useful to keep them
    data['text'] = data["text"].apply(lambda sentence: sentence.replace("can\'t", "can not"))
    data['text'] = data["text"].apply(lambda sentence: sentence.replace("n\'t", " not"))
    data['words'] = data["text"].apply(lambda sentence:  "".join((char if char.isalpha() else " ") for char in sentence).lower().split() )
    print("Tokenization done")

    # getting rid off stopwords
    print("Removing stopwords ...")
    stopwords = nltk.corpus.stopwords.words("english")
    stopwords.remove("not")
    data['words'] = data["words"].apply(lambda words : [word for word in words if word not in stopwords])
    print("Stopwords removed")

    # stemming the words with a Porter Stemmer
    print("Starting stemming ...")
    stemmer = nltk.porter.PorterStemmer()
    data['stems'] = data["words"].apply(lambda words : [stemmer.stem(word) for word in words])
    print("Stemming done")
    
    # performing word embedding
    print("Starting word embedding ...")
    data['words_embedded'] = data['stems'].apply(get_word_embeddings)
    print("Word embedding done")
    
    # averaging the word embedding for a given text
    data['avg_embedding'] = data['words_embedded'].apply(lambda x: np.mean(x, axis =0))
    
    # saving polarisation appart
    print("Starting final formatting of the data ...")
    y = pd.get_dummies(data['sentiment'])

    # transforming the aspect data into dummies
    data = pd.get_dummies(data, columns = ['subject'])

    # getting rid of unnecessary data
    data = data[['avg_embedding',
                 'subject_AMBIENCE#GENERAL', 'subject_DRINKS#PRICES',
                  'subject_DRINKS#QUALITY', 'subject_DRINKS#STYLE_OPTIONS',
                  'subject_FOOD#PRICES', 'subject_FOOD#QUALITY',
                  'subject_FOOD#STYLE_OPTIONS', 'subject_LOCATION#GENERAL',
                  'subject_RESTAURANT#GENERAL', 'subject_RESTAURANT#MISCELLANEOUS',
                  'subject_RESTAURANT#PRICES', 'subject_SERVICE#GENERAL']]

    for i in range(300):
        data["avg_embedding" + '_' + str(i)] = data["avg_embedding"].apply(lambda x: x[i])
    data.drop(["avg_embedding"], axis = 1, inplace = True)

    X = data.values
    # y = y['positive']*1 + y['negative']*-1
    print('Data formated')
    
    return X, y



class Classifier:
    """Le Classifier"""
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        # Loading the training data
        X_train, y_train = formatting_data(trainfile)
        #self.X_train = X_train
        #self.y_train = y_train
        
        self.labels = y_train.columns
        
        print("Starting fitting the model...")
        # fitting a Feed Forward Neural Network
        model = Sequential()
        reduce = ReduceLROnPlateau(monitor="val_loss", patience = 10, factor=.5, verbose=1)
        early  = EarlyStopping(monitor = "val_loss", patience = 100, verbose = 1)
        
        model.add(Dense(300, input_shape = (X_train.shape[1], ), activation ='linear'))
        model.add(Dropout(.25))
        model.add(Dense(150, activation = 'linear'))
        model.add(Dense(100, activation = 'relu'))
        model.add(Dense(50, activation = 'relu'))
        model.add(Dense(20, activation = 'relu'))
        model.add(Dense(3, activation = 'softmax'))
        model.summary()
        model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ["accuracy"])
        model.fit(X_train, y_train, 
                  validation_split = .3, 
                  epochs = 500, batch_size = 128, 
                  verbose = 0, 
                  callbacks=[reduce, early])
        self.model = model
        print("Training done")
        
        
    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        
        X_dev, y_dev = formatting_data(datafile)
        predictions = self.model.predict_classes(X_dev)
        
        predicted_labels = [self.labels[i] for i in predictions]
        return predicted_labels