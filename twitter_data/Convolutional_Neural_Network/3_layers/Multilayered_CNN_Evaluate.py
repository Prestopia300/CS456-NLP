import string 
import re
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from collections import Counter

# load doc in to memory 
def load_doc(filename):
    # open the file as read only 
    file=open(filename,'r', encoding="utf8")
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    # print("cleaned text : " + tokens)
    return tokens


# load all docs in a directory
def process_docs(directory, vocab, is_train):
    documents = list()    
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_train and filename.startswith('cv59'):
            continue
        if is_train and filename.startswith('cv6'):
            continue
        if not is_train and not (filename.startswith('cv59') or filename.startswith('cv6')):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents

# load and clean a dataset
def load_clean_dataset(vocab, is_train):
    # load documents
    neg = process_docs('txt_sentoken/neg', vocab, is_train)
    pos = process_docs('txt_sentoken/pos', vocab, is_train)
    docs = neg + pos
    # prepare labels
    labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
    return docs, labels

# fit a tokenizer (word -> index dictionary) (makes word_index dictionary)
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
    # integer encode (Transforms word in texts -> to corresponding integer value from the word_index dictionary) 
    encoded = tokenizer.texts_to_sequences(docs)
    # pad sequences (make each integer sequence the same length, by adding 0's to the end to increase len to len(max_doc)) 
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded

# classify are view as negative or positive
def predict_sentiment(review, vocab, tokenizer, max_length, model):
    # clean review
    line = clean_doc(review, vocab)
    # encode and pad review
    padded = encode_docs(tokenizer, max_length, [line])
    # predict sentiment
    yhat = model.predict([padded, padded, padded], verbose=0)
    # retriev epredicted percentage and label 
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return(1-percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'


vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
# load training data
train_docs, ytrain = load_clean_dataset(vocab, True)
test_docs, ytest = load_clean_dataset(vocab, False)
# create the tokenizer
tokenizer = create_tokenizer(train_docs)
# define the vocabulary size
vocab_size = len(tokenizer.word_index) + 1 # vocab size + 1 for unknown works
print('Vocabulary size: %d' % vocab_size)
# calculate the maximum sequence length
# to call texts_to_sequences, we need to ensure all documents have the same length
# pad all reviews to the length of the longest training review
max_length = max([len(s.split()) for s in train_docs])
print('Maximum length: %d' % max_length)
# encode data
Xtrain = encode_docs(tokenizer, max_length, train_docs)
Xtest = encode_docs(tokenizer, max_length, test_docs)
# load the model
model = load_model('model.h5')
# evaluate model on training dataset
_, acc = model.evaluate([Xtrain, Xtrain, Xtrain], ytrain, verbose=0)
print('Train Accuracy: %f' % (acc*100))
# evaluate model on test dataset
_, acc = model.evaluate([Xtest, Xtest, Xtest], ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))

# ########################## Test Reivew Evaluation ###########################


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
# num_neg = 0
# num_accurate = 0
# for review in pos_reviews:
#     percent, sentiment = predict_sentiment(review, vocab, tokenizer, max_length, model)
#     print('Review: [%s]\nSentiment: %s (%.3f%%)' % (review, sentiment, percent*100))
#     if sentiment == 'POSITIVE':
#         num_pos += 1
#         num_accurate += 1
#     if sentiment == 'NEGATIVE':
#         num_neg += 1


# for review in neg_reviews:
#     percent, sentiment = predict_sentiment(review, vocab, tokenizer, max_length, model)
#     print('Review: [%s]\nSentiment: %s (%.3f%%)' % (review, sentiment, percent*100))
#     if sentiment == 'NEGATIVE':
#         num_neg += 1
#         num_accurate += 1
#     if sentiment == 'POSITIVE':
#         num_pos += 1

# print('num pos : ' + str(num_pos))
# print('num neg : ' + str(num_neg))
# total_test = len(pos_reviews)+len(neg_reviews)
# print('percent accurate : ' + str(num_accurate) + '/' + str(total_test) )