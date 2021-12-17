# Problem 1 Unigram Model
# # Program to get unique words from a given file
# create dictionary, with 80% of sentences for training
# split sentences into list of words, convert each to lowercase, before storing in the dictionary << DONE

import os
import re
import string
from nltk import sent_tokenize

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

# takes in file, outputs word dictionary and total count
def read_words(filename):
    if os.path.isfile(filename) is not True:
        return None,None
    
    filename = 'doyle_Bohemia.txt'
    file = open(filename, encoding='utf-8')
    text = file.read()
    file.close()

    training_dict ={}
    training_count = 0
    unigram_dict = {}
    unigram_count = 0
    
    sentences =  sent_tokenize(text)

    eighty_percent = int(0.8*len(sentences))
    iterator = 0
    for sentence in sentences:
        if iterator < eighty_percent:
            words = clean_words(sentence)
            for word in words:
                if not word in training_dict:
                    training_dict[word] = 1
                else:
                    training_dict[word] +=1
                training_count +=1
        else:
            words = clean_words(sentence)
            for word in words:
                if not word in unigram_dict:
                    unigram_dict[word] = 1
                else:
                    unigram_dict[word] +=1
                unigram_count +=1
        iterator += 1
    
    return training_dict, training_count, unigram_dict, unigram_count

 
def write_prob(training_dict,training_count,unigram_dict, unigram_count, file_name):
    with open(file_name,'w') as outfile:
        # for each word in dictionary
        for word,count in unigram_dict.items():
            # write word = count/totalwords\n
            # {1:.xf}, where x is the number of precision of the decimal
            # outfile.write('{0} = {1:.8f}\n'.format(word,float(count)/total_words))
            if word in training_dict:
                outfile.write('{0} = {1:.8f}\n'.format(word,float(training_dict[word])/training_count))
            else:
                outfile.write('{0} = {1:.8f}\n'.format(word,float(0)/training_count))


def unigram_eval(input_file_name, output_file_name):
    training_dict, training_count, unigram_dict, unigram_count = read_words(input_file_name)
    
    # if os.path.isfile(input_file_name) is not True:
    #     return None
    
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

                for word in words:
                    if word in training_dict:
                        wordprob = float(training_dict[word] / training_count)
                        sentprob *= wordprob
                    else:
                        sentprob *= 0
                sent_numb += 1
                total_prob += float(sentprob)
                outfile.write('sentence ' + str(sent_numb) + ' probability = {0:.8f}\n'.format(float(sentprob)))
            iterator += 1
        outfile.write('\ntotal sentence ' + str(sent_numb) + ' probability = {0:.20f}\n'.format(float(total_prob)))

def main():
    input_file_name='doyle_Bohemia.txt'
    output_file_name='unigram_probs.txt'
    training_dict, training_count, unigram_dict, unigram_count = read_words(input_file_name)
    if training_dict is not None:
        write_prob(training_dict,training_count,unigram_dict, unigram_count,output_file_name)
        print(output_file_name + ' updated successfully')
    else:
        print('Unable to read file : ' + input_file_name)

    # <<<<<<< unigram_eval.txt 
    output_file_name = 'unigram_eval.txt'
    if training_dict is not None:
        unigram_eval(input_file_name, output_file_name)
        print(output_file_name + ' updated successfully')
    else:
        print('Unable to read file : ' + input_file_name)

main()