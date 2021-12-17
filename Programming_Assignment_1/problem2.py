import os
import re
import string
import nltk
from nltk import sent_tokenize
from nltk.util import ngrams 


# cleans text corpus 
def clean_words(text):
    text = text.replace('“', '')
    text = text.replace('”', '')
    text = text.replace('‘', '')
    text = text.replace('’', '')
    text = text.replace('—', '-')
    #text = text.replace('-', ' ')
    text = text.replace('\n', ' ')

    text = text.split()
    text = [ w.strip(string.punctuation).lower() for w in text ]

    return text


# takes in file, outputs word dictionary and total count - fpr a word_pair
# calculate the frequency of each individual word
def read_words(filename):
    if os.path.isfile(filename) is not True:
        return None
    
    filename = 'doyle_Bohemia.txt'
    file = open(filename, encoding='utf-8')
    text = file.read()
    file.close()

    training_bigram_dict ={}
    training_bigram_count = 0

    training_word_dict = {}
    training_word_count = 0

    testing_bigram_dict = {}
    testing_bigram_count = 0

    testing_word_dict = {}
    testing_word_count = 0


    sentences =  sent_tokenize(text)

    eighty_percent = int(0.8*len(sentences))
    iterator = 0
    
    # this creates a frequence diagram for word_pairs
    for sentence in sentences:
        if iterator < eighty_percent:
            # individual word (training_word_dict)
            words = clean_words(sentence)
            
            for word in words:
                if not word in training_word_dict:
                    training_word_dict[word] = 1
                else:
                    training_word_dict[word] += 1
                training_word_count += 1 
            
            # bigram (training_bigram_dict)
            bigram = list(nltk.bigrams(words))

            for word_pair in bigram:
                if not word_pair in training_bigram_dict:
                    training_bigram_dict[word_pair] = 1
                else:
                    training_bigram_dict[word_pair] +=1
                training_bigram_count +=1
        else:
            # individual word (testing_word_dict)
            words = clean_words(sentence)
            
            for word in words:
                if not word in testing_word_dict:
                    testing_word_dict[word] = 1
                else:
                    testing_word_dict[word] += 1
                testing_word_count += 1
            
            # bigram (testing_bigram_dict)
            bigram = list(nltk.bigrams(words))

            for word_pair in bigram:
                if not word_pair in testing_bigram_dict:
                    testing_bigram_dict[word_pair] = 1
                else:
                    testing_bigram_dict[word_pair] +=1
                testing_bigram_count +=1

        iterator += 1    
    # bigram, word
    return training_bigram_dict, training_bigram_count, testing_bigram_dict, testing_bigram_count, training_word_dict, training_word_count, testing_word_dict, testing_word_count

def write_prob(training_b_dict, testing_b_dict, training_w_dict, file_name):
    # training_dict, training_count, bigram_dict, bigram_count

    with open(file_name,'w') as outfile:
        # for each word in dictionary
        for word_pair,count in testing_b_dict.items():
            # write word = count/totalwords\n
            # {1:.xf}, where x is the number of precision of the decimal
            if word_pair in training_b_dict.items():
                # number of word_pair's in training / number of first word 
                outfile.write('{0} = {1:.8f}\n'.format(word_pair, float( training_b_dict[word_pair] / training_w_dict[word_pair[0]] ))) # <<<<<<<<<<< what is V ???
            else:
                outfile.write('{0} = {1:.8f}\n'.format(word_pair, float(0) ))


def bigram_eval(input_file_name, output_file_name):   
    training_b_dict, training_b_count, testing_b_dict, testing_b_count, training_w_dict, training_w_count, testing_w_dict, testing_w_count = read_words(input_file_name)

    if os.path.isfile(input_file_name) is not True:
        return None
    
    file = open(input_file_name, encoding='utf-8')
    text = file.read()
    file.close()

    sentences =  sent_tokenize(text)

    eighty_percent = int(0.8*len(sentences))
    iterator = 0
    sent_numb = 0
    total_prob = float(0)
    # sentence prob = multiplying the probabilities of each word together
    with open(output_file_name,'w') as outfile:
        for sent in sentences:
            if iterator >= eighty_percent:
                sentprob = float(1)
                words = clean_words(sent)

                # bigram (testing_bigram_dict)
                bigram = list(nltk.bigrams(words))

                for word_pair in bigram:
                    if word_pair in training_b_dict.items():
                        word_pair_prob = float(training_b_dict[word_pair] / training_w_dict[word_pair[0]])
                        sentprob *= word_pair_prob
                    else:
                        sentprob *= 0
                sent_numb += 1
                total_prob += float(sentprob)
                outfile.write('sentence ' + str(sent_numb) + ' probability = {0:.8f}\n'.format(float(sentprob)))
            iterator += 1
        outfile.write('\ntotal sentence ' + str(sent_numb) + ' probability = {0:.20f}\n'.format(float(total_prob)))


def main():
    input_file_name = 'doyle_Bohemia.txt'
    output_file_name = 'bigram_probs.txt'
    # training_dict, training_count, bigram_dict, bigram_count 
    training_b_dict, training_b_count, testing_b_dict, testing_b_count, training_w_dict, training_w_count, testing_w_dict, testing_w_count = read_words(input_file_name)
    
    if training_b_dict is not None:
        write_prob(training_b_dict, testing_b_dict, training_w_dict, output_file_name)
        print(output_file_name + ' updated successfully')
    else:
        print('Unable to read file : ' + input_file_name)


    # <<<<<<< bigram_eval.txt 
    output_file_name = 'bigram_eval.txt'
    if training_b_dict is not None:
        bigram_eval(input_file_name, output_file_name)
        print(output_file_name + ' updated successfully')
    else:
        print('Unable to read file : ' + input_file_name)


main()