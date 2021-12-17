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


def write_transition_prob(pair_frequency, tag_frequency, pair_list):
    # Goal:
    # calculate Probability(tagi|tagi-1) = Count(tagi-1, tagi) / Count(tagi-1)

    # find tag pair frequencies
    tag_pairs = {}
    i = 0
    # start to first tag
    tup = ('<s>', pair_list[0][1])
    tag_pairs[tup] = 1

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
            if curr_tag2 != '<s>':
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

def viterbi(emission_probs, transition_probs, tag_frequency, sentence): # pair_frequency, tag_frequency, , pair_list):
    # ! Generally replace products of probabilities with sums of log probabilities
    # number = Log(word|Curr_POS) + Max[Score(Prev_POS) + Log(Curr_POS|Prev_POS) for all Prev_POS to CURR_POS transitions]


    T = len(tag_frequency)

    words = sentence.split()
    sent_len = len(words)

    word_one = words[0]
    Score = {}
    BackPtr = {}

    # Check if sentence tags are computable
    can_compute = True
    for word in words:
        if not word in emission_probs:
            can_compute = False

    if can_compute == True:

        # Initialization Step
        for curr_tag in tag_frequency:
            # Pr(Word1| Tagt)
            em_prob = 0
            for item in emission_probs[word_one]:
                tag, prob = item
                if tag == curr_tag:
                    em_prob = prob

            # Pr(Tagt| φ)
            start_tag_prob = 0
            for item in transition_probs['<s>']:
                tag2, prob = item
                if tag2 == curr_tag:
                    start_tag_prob = prob
            
            Score[(curr_tag,0)] = em_prob * start_tag_prob
            BackPtr[(curr_tag,0)] = '<s>'

        # Iteration Step
        for w in range(1, sent_len):
            for curr_tag in tag_frequency:
                
                # Pr(Wordw| Tagt)
                em_prob = 0
                for item in emission_probs[words[w]]:
                    tag, prob = item
                    if tag == curr_tag:
                        em_prob = prob
                
                # MAXj=1,T(Score(j, w-1) * Pr(Tagt| Tagj)) 

                max_prob = -float("inf")
                max_index = ''
                for tagj in tag_frequency:
                    
                    # Pr(Tagt| Tagj))
                    tagj_prob = 0
                    for item in transition_probs[tagj]:
                        tag2, prob = item
                        if tag2 == curr_tag:
                            tagj_prob = prob

                    curr_prob = Score[(tagj, w-1)] * tagj_prob

                    # update max                
                    if curr_prob > max_prob:
                        max_prob = curr_prob
                        max_index = tagj
                
                # Score(t, w) = Pr(Wordw| Tagt) *MAXj=1,T(Score(j, w-1) * Pr(Tagt| Tagj))    
                Score[(curr_tag, w)] = em_prob * max_prob
                BackPtr[(curr_tag,w)] = max_index

        # Sequence Identification
        # finds ending tag, then backtracks 
        Seq = {}
        # Seq(sent_len ) = t that maximizes Score(t,sent_len )
        max_score = -float('inf')
        max_prob_tag = 'U' # max_prob_tag defaults to U (unknown)

        for curr_tag in tag_frequency:
            if ((curr_tag,sent_len-1) in Score):
                if (Score[(curr_tag,sent_len-1)] > max_score):
                    max_score = Score[(curr_tag,sent_len-1)]
                    max_prob_tag = curr_tag

        Seq[words[sent_len-1]] = max_prob_tag

        for w in reversed(range(sent_len-1)):
            #curr Seq =  BP(nxt_tab, nxt_word)
            # print(str(Seq[w + 1]) + ', ' + str(w + 1)) # (N,2) this means that tab should be the actual tab, not a number
            Seq[words[w]] = BackPtr[Seq[words[w + 1]], w + 1]
        
        # Seq[0] = BackPtr[Seq[0], 0] # move other w up one for this to work
        return Seq
    else:
        Seq = {}
        for word in words:
            if not word in emission_probs:
                Seq[word] = 'Unknown'
            # else, compute emission prob
            else:
                max_prob = -float('inf')
                max_tag = ''
                # tup = ('<s>', 0.1)
                # word_probs_dict[word] = [tup]
                for item in emission_probs[word]:
                    tag, prob = item
                    if prob > max_prob:
                        max_prob = prob
                        max_tag = tag
                Seq[word] = max_tag

               
        return Seq

    # Perhaps Use these as a sum instead of current as product
    # number = math.log2()
    # math.log2(number)

def main():
    input_file_name='Klingon_Train.txt'
    pair_list = read_words(input_file_name)

    pair_frequency, tag_frequency = write_frequencies(pair_list)

    # EMISSION
    emission_probs = write_emission_prob(pair_frequency, tag_frequency, pair_list)
    
    # print('Emission Probability Table')
    # print('----------------------------')
    # for pair in sorted (emission_probs.keys()):
    #     for tup in emission_probs[pair]:
    #         tag = tup[0]
    #         prob = tup[1]
    #         # tag, prob = tup
    #         print(str(pair) + '/' + str(tag) + ' --> ' + str(prob))
    #     print()
        
    # print('----------------------------\n\n')
    
    # TRANSITION
    tag_pairs, transition_probs = write_transition_prob(pair_frequency, tag_frequency, pair_list)

    # print('Transition Probability Table')
    # print('----------------------------')
    # for first in sorted (transition_probs.keys()):
    #     # print(item + ' ' + str(transition_probs[item]))
    #     for second, prob in transition_probs[first]:
    #         print('' + str(first) + ' to ' + str(second) + ' --> ' + str(prob) )
    #     print()
    # print('----------------------------\n\n')

    # VITERBI
    sentence = 'tera’ngan legh yaS'
    # sentence = 'taH qIp puq'
    viterbi_lst = viterbi(emission_probs, transition_probs, tag_frequency, sentence)

    print('Viterbi Probability Table')
    print('----------------------------')
    for item in viterbi_lst:
        print(item + '/' + viterbi_lst[item])
    print('----------------------------')

main()