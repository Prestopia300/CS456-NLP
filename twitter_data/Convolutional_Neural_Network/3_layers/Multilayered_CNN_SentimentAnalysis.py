import string 
import re
from os import listdir
from nltk.corpus import stopwords
from pickle import dump

from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate

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
    
    # # split into tokens by white space
    # tokens = doc.split()
    # # prepare regex for char filtering
    # re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # # remove punctuation from each word
    # tokens = [re_punc.sub('', w) for w in tokens]
    # # remove remaining tokens that are not alphabetic
    # tokens = [word for word in tokens if word.isalpha()]
    # # filter out stopwords 
    # stop_words = set(stopwords.words('english'))
    # tokens = [w for w in tokens if not w in stop_words]
    # # filter out short tokens 
    # tokens = [word for word in tokens if len(word) > 1]
    # tokens = ' '.join(tokens)
    # return tokens

# load all docs in a directory
def process_docs(directory, vocab, is_train):
    documents = list()    
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any tweets in the test set 
        if is_train and (filename.startswith('cv59') or filename.startswith('cv6')):
            continue
        # skip any tweets in the train set
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
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels

# fit a tokenizer (word -> index dictionary) (makes word_index dictionary)
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def encode_docs(tokenizer, length, lines):
    # integer encode (Transforms word in texts -> to corresponding integer value from the word_index dictionary) 
    encoded = tokenizer.texts_to_sequences(lines)
    # pad sequences (make each integer sequence the same length, by adding 0's to the end to increase len to len(max_doc)) 
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded

# define the model
def define_model(length, vocab_size):
    dimentions = 50
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, dimentions)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # channel2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, dimentions)(inputs2)
    conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # channel 3
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size, dimentions)(inputs3)
    conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    # merge
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize 
    model.summary()
    plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model


vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())

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
model = define_model(max_length, vocab_size)


model.fit([Xtrain,Xtrain,Xtrain], array(ytrain), epochs=10, batch_size=16) # changed epochs from 7 to 10


# fit network
# model.fit(Xtrain, ytrain, epochs=15, verbose=2) # 10 epochs = 10 passes through the training data
# save the model
model.save('model.h5')