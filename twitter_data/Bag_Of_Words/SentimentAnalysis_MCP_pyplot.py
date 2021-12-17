# This code predicts the sentiment of new reviews, using the following steps to prepare the test data:
# loading the text, 
# cleaning the document, 
# filtering tokens by the chosen vocabulary, 
# converting the remaining tokens to a line, 
# encoding it using the Tokenizer,
# and making a prediction. 

import string
import re
from os import listdir
import numpy as np
from numpy import array

from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model 
from keras.models import Sequential
from keras.layers import Dense
from pandas import DataFrame
from matplotlib import pyplot

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename,'r', encoding="utf8")
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc):
    #split into tokens by white space 
    tokens = doc.split()
    # prepare regex for char filtering 
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word 
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic 
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stopwords 
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens 
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# load doc, clean and return line of tokens 
def doc_to_line(filename, vocab):
    # load the doc 
    doc = load_doc(filename)
    # clean doc 
    tokens = clean_doc(doc)
    # filter by vocab 
    tokens = [w for w in tokens if w in vocab]
    if len(tokens) <= 1:
        print("NO WORDS IN TWEET!" + tokens[0] + filename + "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n") # 41.txt has only 1 word : 'one'

    return ' '.join(tokens)

# load all docs in a directory 
def process_docs(directory, vocab, is_train):
    lines = list()
    # walk through all files in the folder 
    count = 0
    for filename in listdir(directory):
        # skip any reviews in the test set 
        if is_train and filename.startswith('cv59'):
            continue
        if is_train and filename.startswith('cv6'):
            continue
        if not is_train and not (filename.startswith('cv59') or filename.startswith('cv6')):
            continue
        # if not is_train and not filename.startswith('cv6'):
        #     continue
        count += 1
          
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        line = doc_to_line(path, vocab)
        # add to list 
        lines.append(line)
    print("count : " + str(count))
    return lines

# load and clean a data set 
def load_clean_dataset(vocab, is_train):
    # load documents 
    neg = process_docs('txt_sentoken/neg', vocab, is_train)
    pos = process_docs('txt_sentoken/pos', vocab, is_train)
    docs = neg + pos
    # prepare labels
    labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])



    print(labels)
    return docs, labels

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# define the model
def define_model(n_words):
    # define network
    model = Sequential()
    model.add(Dense(50, input_shape=(n_words,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

# prepare bag-of-words encoding of docs
def prepare_data(train_docs, test_docs, mode_in):
    # create to tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_docs)
    # encode training data set
    Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode_in)
    # encode training data set
    Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode_in)
    return Xtrain, Xtest

# evaluate a neural network model
def evaluate_mode(Xtrain, ytrain, Xtest, ytest, NoOfNeurons, NoOfEpoc):
    scores = list()
    # scores = np.array(scores)
    n_words = Xtest.shape[1]
    # define network
    model = Sequential()
    model.add(Dense(NoOfNeurons, input_shape=(n_words,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(Xtrain, ytrain, epochs=NoOfEpoc, verbose=2)
    # evaluate

    loss, acc = model.evaluate(Xtest, ytest, verbose=0)
    scores.append(acc)
    # print("%d accuracy: %s" % ((i+1), acc))
    return scores   


# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
# load all reviews 
train_docs, ytrain = load_clean_dataset(vocab, True)
test_docs, ytest = load_clean_dataset(vocab, False)
# reun experiment

# print(train_docs)
print("--------------------------------------------------------------------------------")
print(test_docs)

# modes = ['binary', 'count', 'tfidf', 'freq']
# results = DataFrame()
# for mode in modes:
#     # prepare data for mode
#     Xtrain, Xtest = prepare_data(train_docs, test_docs, mode)
#     # evaluate model on data for mode
#     results[mode] = evaluate_mode(Xtrain, ytrain, Xtest, ytest, 50)
# # summarize results
# print(results.describe())
# # plot results
# results.boxplot()
# pyplot.show()


# Number of Neurons in Hidden Layer (Adapted from above Code)

NoOfEpoc = [5,10,15,20]
NoOfNeurons = [50, 100, 150, 200]

for EpocNo in NoOfEpoc:

    results = DataFrame()
    for NeronNo in NoOfNeurons:
        # prepare data for mode
        Xtrain, Xtest = prepare_data(train_docs, test_docs, 'tfidf')
        # evaluate model on data for mode
        results[NeronNo] = evaluate_mode(Xtrain, ytrain, Xtest, ytest, NeronNo, EpocNo)
    # summarize results
    print(results.describe())
    # plot results
    results.boxplot()
    pyplot.show()