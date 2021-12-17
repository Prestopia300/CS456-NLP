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
import numpy
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
    return ' '.join(tokens)

# load all docs in a directory 
def process_docs(directory, vocab):
    lines = list()
    # walk through all files in the folder 
    for filename in listdir(directory):
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        line = doc_to_line(path, vocab)
        # add to list 
        lines.append(line)
    return lines

# load and clean a data set 
def load_clean_dataset(vocab):
    # load documents 
    neg = process_docs('txt_sentoken/neg', vocab)
    pos = process_docs('txt_sentoken/pos', vocab)
    docs = neg + pos
    # prepare labels
    labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
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

# classify a review as negative or positive
def predict_sentiment(review, vocab, tokenizer, model):
    # clean
    tokens = clean_doc(review)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    # convert to line
    line = ' '.join(tokens)
    # encode
    encoded = tokenizer.texts_to_matrix([line], mode='tfidf')
    # predict sentiment
    yhat = model.predict(encoded, verbose=0)
    # retrieve predicted percentage and label
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return (1-percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'


# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
# load all reviews 
train_docs, ytrain = load_clean_dataset(vocab)
test_docs, ytest = load_clean_dataset(vocab)
# create the tokenizer
tokenizer = create_tokenizer(train_docs)
# encode data
# Xtrain, Xtest = prepare_data(train_docs, test_docs, mode)
Xtrain = tokenizer.texts_to_matrix(train_docs, mode='tfidf')
Xtest = tokenizer.texts_to_matrix(test_docs, mode='tfidf')
# define network
n_words = Xtrain.shape[1]
model = define_model(n_words)
# fit network
model.fit(Xtrain, ytrain, epochs=17, verbose=2)

_, acc = model.evaluate(Xtrain, ytrain, verbose=0)
print('Train Accuracy: %f' % (acc*100))
# evaluate model on test dataset
_, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))


# pos_reviews_path = []
# neg_reviews_path = []

# for i in range(600,650):
#     pos_path = 'txt_sentoken/pos/' + "cv{}.txt".format(i)
#     pos_reviews_path.append(pos_path)
#     neg_path = 'txt_sentoken/neg/' + "cv{}.txt".format(i)
#     neg_reviews_path.append(neg_path)

# # pos text
# pos_reviews = []
# for path in pos_reviews_path:
#     pos_file_text = load_doc(path)
#     pos_reviews.append(pos_file_text)

# # neg text
# neg_reviews = []
# for path in neg_reviews_path:
#     neg_file_text = load_doc(path)
#     neg_reviews.append(neg_file_text)

# num_pos = 0
# for review in pos_reviews:
#     percent, sentiment = predict_sentiment(review, vocab, tokenizer, model)
#     print('Review: [%s]\nSentiment: %s (%.3f%%)' % (review, sentiment, percent*100))
#     if sentiment == 'POSITIVE':
#         num_pos += 1

# num_neg = 0
# for review in neg_reviews:
#     percent, sentiment = predict_sentiment(review, vocab, tokenizer, model)
#     print('Review: [%s]\nSentiment: %s (%.3f%%)' % (review, sentiment, percent*100))
#     if sentiment == 'NEGATIVE':
#         num_neg += 1

# print('num pos : ' + str(num_pos))
# print('num neg : ' + str(num_neg))