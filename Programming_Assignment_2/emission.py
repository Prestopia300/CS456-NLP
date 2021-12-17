# Problem 1 Unigram Model
# # Program to get unique words from a given file
# create dictionary, with 80% of sentences for training
# split sentences into list of words, convert each to lowercase, before storing in the dictionary << DONE

import os
import re
import string
from nltk import sent_tokenize
from nltk import pos_tag

# cleans txt input to a list
def read_words(filename):
    if os.path.isfile(filename) is not True:
        return None
    
    
    file = open(filename, encoding='utf-8')
    lines = file.readlines()
    file.close()

    pair_list = []
    for line in lines:
        splitted = line.split()
        for item in splitted:
            pair = item.split("/")
            tup = (pair[0], pair[1])
            pair_list.append(tup)

    return pair_list


def write_frequencies(pair_list):
    # Goal:
    # calculate frequency of Count(word,tag)
    # calculate frequency of Count(tag)

    pair_frequency = {}
    tag_frequency = {}

    # pair_frequency
    for item in pair_list:
        if not item in pair_frequency:
            pair_frequency[item] = 1
        else:
            pair_frequency[item] += 1

    # tag_frequency
    tag_frequency['<s>'] = 1
    for item in pair_list:
        if not item[1] in tag_frequency:
            tag_frequency[item[1]] = 1
        else:
            tag_frequency[item[1]] += 1

    return pair_frequency, tag_frequency


def write_emission_prob(pair_frequency, tag_frequency, pair_list):
    # Goal:
    # calculate Probability(word|tag) = Count(word,tag) / Count(tag)
    # the output will be output for each possible word, and for each word, each possible POS tag, with its probablility

    word_probs_dict = {}

    for pair in pair_list:
        word, tag = pair
        
        tup = ('<s>', 0.1)
        word_probs_dict[word] = [tup]

        for curr_tag in tag_frequency:
            prob = 0
            if curr_tag == tag:
                prob = (pair_frequency[pair]) / (tag_frequency[tag])
            else:
                prob = 0
            prob += 0.1
            tup = (curr_tag, prob)

            if not word in word_probs_dict:
                word_probs_dict[word] = [tup]
            else:
                # no tag repititions
                insert = True
                for item in word_probs_dict[word]:
                    if item[0] == tup[0]:
                        insert = False
                if insert == True:
                    word_probs_dict[word].append(tup)

    return word_probs_dict   # emission prob
    

def main():
    input_file_name='Klingon_Train.txt'
    pair_list = read_words(input_file_name)

    pair_frequency, tag_frequency = write_frequencies(pair_list)

    # EMISSION
    emission_probs = write_emission_prob(pair_frequency, tag_frequency, pair_list)
    
    print('Emission Probability Table')
    print('----------------------------')
    for pair in sorted (emission_probs.keys()):
        for tup in emission_probs[pair]:
            tag = tup[0]
            prob = tup[1]
            # tag, prob = tup
            print(str(pair) + '/' + str(tag) + ' --> ' + str(prob))
        print()
        
    print('----------------------------')

main()