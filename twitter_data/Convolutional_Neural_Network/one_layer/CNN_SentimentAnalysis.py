import string 
import re
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

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

def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=max_length)) # vector space can change, we chose 100, max_length is same as padding
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu')) # for CNN
    model.add(MaxPooling1D(pool_size=2))  # reduces CNN output by half
    model.add(Flatten()) # one long 2d vector represents features extracted
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) # sigmoid outputs 0 or 1 for negative or positive reviews
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # this is a binary classification problem
    # summarize defined model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
# load training data
train_docs, ytrain = load_clean_dataset(vocab, True)
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

# define model
model = define_model(vocab_size, max_length)
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2) # 10 epochs = 10 passes through the training data
# save the model
model.save('model.h5')