N-SKIP GRAMS WITH NEGATIVE SAMPLES

Implementation of n-skip grams with negative samples. Sub-sambling is also perfomed in order to reduce the training corpus.

Libraries : 

>> tqdm : makes a progression bar appear when performing a loop
>> json : import/export data as json
>> the other libraries that were in the 

Parameters :

>> winsize : Size of the context => 3 
>> nEmbed : Number of neurons in the hidden layer => 200 (We don't take into account any bias in the layers)
>> Epochs : Number of time the whole dataset is passed thrue the nodes during the training = > 30 (Larger number of epochs was difficult for us due to computation time limitation)
>> Mincount : Minimum number of instances for a word to appear in our training data => 15
>> subSamplingThreshold :Frequency threshold to perform sub sampling according to the threshold on the Word2vec literature
>> Stopwords : List of common stopwords that we exclude for training
>> Stepsize : learning rate  => 0.001 

The code is also available on github : https://github.com/pauldechorgnat/NLP-CS/tree/master/n-skip%20grams%20with%20negative%20samples with some training data (taken from gutemberg.org)


