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
        
        for curr_tag in tag_frequency:
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


def write_transition_prob(pair_frequency, tag_frequency, pair_list):
    # Goal:
    # calculate Probability(tagi|tagi-1) = Count(tagi-1, tagi) / Count(tagi-1)

    # find tag pair frequencies
    tag_pairs = {}
    i = 0
    for pair in pair_list:
        if (i != 0):
            # one tag to another tag
            tup = (pair_list[i-1][1], pair_list[i][1])
            if not tup in tag_pairs:
                tag_pairs[tup] = 1
            else:
                tag_pairs[tup] += 1
        i += 1
    
    # calculate probabilities
    transition_probs = {}

    for curr_tag1 in tag_frequency:
        for curr_tag2 in tag_frequency:
            pair = (curr_tag1, curr_tag2)
            if pair in tag_pairs:
                # print(str(tag_pairs[pair]) + ' / ' + str(tag_frequency[curr_tag1]) + ' 0.1')
                prob = (tag_pairs[pair]) / (tag_frequency[curr_tag1])
            else:
                prob = 0
            prob += 0.1
    
            tup = (curr_tag2, prob)

            # cannot have both first and second
            if not curr_tag1 in transition_probs:
                # not first (second yes or not)
                transition_probs[curr_tag1] = [tup]
            else:
                # yes first not second
                if not tup in transition_probs[curr_tag1]:
                    transition_probs[curr_tag1].append(tup)

            for item in transition_probs:
                transition_probs[item].sort()
            
    return tag_pairs, transition_probs


def viterbi(sentence, word_probs_dict): # pair_frequency, tag_frequency, , pair_list):
    # T = len(tag_frequency)
    # W = len(word_probs_dict)
    # Score = {}
    # BackPtr = {}

    # i = 1
    # for word in word_probs_dict:
    #     tag, prob = word_probs_dict[word]
    #     Score[(i, 1)] = prob * tag_frequency[tag]
    #     BackPtr[(i, 1)] = 0
    #     i += 1

    # w = 2
    # t = 1
    
    # for tag in tag_frequency:

    words = sentence.split()

    # looks at each words probability individually (not really viterbi)
    viterbi_lst = []
    for word in words:
        if not word in word_probs_dict:
            not_in_words_from_testing = ('Unknown', -float('inf'))
            viterbi_lst.append(not_in_words_from_testing)
        else:
            word_prob_lst = word_probs_dict[word]
            max_prob = ('Unknown', -float('inf'))
            for tag, prob in word_prob_lst:
                if prob > max_prob[1]:
                    max_prob = (tag, prob)
            viterbi_lst.append(max_prob)
    return viterbi_lst
    
    



def main():
    input_file_name='Klingon_Train.txt'
    pair_list = read_words(input_file_name)

    # for item in pair_list:
    #     print(item)

    pair_frequency, tag_frequency = write_frequencies(pair_list)

    # for pair in pair_list:
    #     print(str(pair) + ' ' + str(pair_frequency[pair]))

    # for tag in tag_frequency:
    #     print(tag + ' ' + str(tag_frequency[tag])) 

    # EMISSION
    emission_probs = write_emission_prob(pair_frequency, tag_frequency, pair_list)
    
    print('Emission Probability Table')
    print('----------------------------')
    for pair in sorted (emission_probs.keys()):
        for tup in emission_probs[pair]:
            tag = tup[0]
            prob = tup[1]
            # tag, prob = tup
            print(str(pair) + ' : ' + str(tag) + ' prob --> ' + str(prob))
        print()
        
    print('----------------------------\n\n')
    
    # TRANSITION
    tag_pairs, transition_probs = write_transition_prob(pair_frequency, tag_frequency, pair_list)

    print('Transition Probability Table')
    print('----------------------------')
    for first in sorted (transition_probs.keys()):
        # print(item + ' ' + str(transition_probs[item]))
        for second, prob in transition_probs[first]:
            print('Pair: ' + str(first) + ' and ' + str(second) + ' prob --> ' + str(prob) )
        print()
    print('----------------------------')

    # VITERBI
    sentence = 'tera’ngan legh yaS'
    viterbi_lst = viterbi(sentence, emission_probs)

    print('Viterbi Probability Table')
    print('----------------------------')
    for item in viterbi_lst:
        print(item[0])
    print('----------------------------')

main()