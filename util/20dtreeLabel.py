# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:28:35 2015

@author: wangwenya
"""

"""
create tree structures from raw parses for training sentences
accumulate vocabulary
ignore lemmatization
differentiate beginning and inside of aspects
"""


from dtree_util import *
import gen_util as gen
import sys, cPickle, random, os
from numpy import *

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
#from nltk.tokenize import word_tokenize
'''
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None
    
wordnet_lemmatizer = WordNetLemmatizer()
'''

f = open('data_semEval/raw_parses_res_new', 'r')
#sentence_file = open('data_semEval/sentence_res', 'r')

f_token_pos = open('data_semEval/uni_token_pos_res_new', 'rb')
token_pos = cPickle.load(f_token_pos)
indice = 0

data = f.readlines()
plist = []
tree_dict = []
vocab = []
rel_list = []

label_file = open('data_semEval/aspectTerm_res_BIO', 'r')
#opinion_positive_file = open('opinion-lexicon/positive-words.txt', 'r')
#opinion_negative_file = open('opinion-lexicon/negative-words.txt', 'r')
#
#opinion_positive = opinion_positive_file.read().splitlines()
#opinion_negative = opinion_negative_file.read().splitlines()
label_sentence = open('data_semEval/addsenti_res_new1.txt', 'r')

for line in data:
    if line.strip():
        rel_split = line.split('(')
        rel = rel_split[0]
        deps = rel_split[1][:-1]
        deps = deps.replace(')','')
        if len(rel_split) != 2:
            print 'error ', rel_split
            sys.exit(0)

        else:
            dep_split = deps.split(',')
            
        if len(dep_split) > 2:
            fixed = []
            half = ''
            for piece in dep_split:
                piece = piece.strip()
                if '-' not in piece:
                    half += piece

                else:
                    fixed.append(half + piece)
                    half = ''

                    #print 'fixed: ', fixed
            dep_split = fixed

        final_deps = []
        for dep in dep_split:
            words = dep.split('-')
            word = words[0]
            ind = int(words[len(words) - 1])

            if len(words) > 2:
                word = '-'.join([w for w in words[:-1]])

            final_deps.append( (ind, word.strip()) )
            
        plist.append((rel,final_deps))

    else:
        max_ind = -1
        for rel, deps in plist:
            for ind, word in deps:
                if ind > max_ind:
                    max_ind = ind

        # load words into nodes, then make a dependency tree
        nodes = [None for i in range(0, max_ind + 1)]
        for rel, deps in plist:
            for ind, word in deps:
                nodes[ind] = word#.lower()

        tree = dtree(nodes)
        
 
        #add code corresponding to customer review data       
        #labels = label_file.readline().strip()
        
        opinion_words = []
        '''
        sentence = sentence_file.readline().strip()
        terms = word_tokenize(sentence)
        pos_review = nltk.pos_tag(terms)
        '''
        
        sequence = token_pos[indice]
        
        '''
        labeled_sent = label_sentence.readline().strip()
        if '##' in labeled_sent:
            opinion_line = labeled_sent.split('##')[1].strip()
            opinions = opinion_line.split(',')
            for opinion in opinions:
                opinion_word = opinion.strip().split(' ')[:-1]
                for word in opinion_word:
                    opinion_words.append(word)
        '''
        '''
        word_pos = {}
        for item in pos_review:
            word_pos[item[0]] = penn_to_wn(item[1])
        '''
        
        '''
        pos_dic = {}
        for pair in sequence:
            pos_dic[pair[0]] = penn_to_wn(pair[1])
        '''
        
        '''    
        #check if the sentence has aspects
        if label[0] != '':
        
            aspects = label[0].split(',')
            for aspect in aspects:
                aspect = aspect.split('[')[0].strip()
                for word in aspect.split(' '):
                    if word != '':
                        aspect_words.append(word.strip())
            
        if len(label) > 3:
            sentiments = label[2:-1]
            for sentiment in sentiments:
                sentiment = sentiment.strip()
                sentiment = sentiment.split(' ')
                
                for word in sentiment[:-1]:
                    opinion_words.append(word)
                
                """
                if sentiment[len(sentiment) - 1] == '-1':
                    for word in sentiment[:-1]:
                        negative_words.append(word)
                else:
                    for word in sentiment[:-1]:
                        positive_words.append(word)
                """
        '''
            
        
        aspect_term = label_file.readline().rstrip()
        labeled_sent = label_sentence.readline().strip()
        #facilitate bio notation
        aspect_BIO = {}
        
        if '##' in labeled_sent:
                opinions = labeled_sent.split('##')[1].strip()
                opinions = opinions.split(',')
                
                for opinion in opinions:
                    op_list = opinion.split()[:-1]
                    if len(op_list) > 1:
                        for ind, term in enumerate(nodes):
                            if term != None:
                                if term == op_list[0] and ind < len(nodes) - 1 and nodes[ind + 1] != None and nodes[ind + 1] == op_list[1]:
                                    tree.get(ind).trueLabel = 3
                                    for i in range(len(op_list) - 1):
                                        if nodes[ind + i + 1] != None and nodes[ind + i + 1] == op_list[i + 1]:
                                            tree.get(ind + i + 1).trueLabel = 4
                                        
                    elif len(op_list) == 1:
                        for ind, term in enumerate(nodes):
                            if term != None:
                                if term == op_list[0] and tree.get(ind).trueLabel == 0:
                                    tree.get(ind).trueLabel = 3
        
        if aspect_term != 'NIL':
            #aspect_term += " "
            #aspects = aspect_term.split(' ')
            aspects = aspect_term.split(',')
            
                        
            #deal with same word but different labels
            for aspect in aspects:
                aspect = aspect.strip()
                #aspect is a phrase
                if ' ' in aspect:
                    aspect_list = aspect.split()
                    for ind, term in enumerate(nodes):
                        if term == aspect_list[0] and ind < len(nodes) - 1 and nodes[ind + 1] == aspect_list[1]:
                            tree.get(ind).trueLabel = 1
                            
                            for i in range(len(aspect_list) - 1):
                                if ind + i + 1 < len(nodes):
                                    if nodes[ind + i + 1] == aspect_list[i + 1]:
                                        tree.get(ind + i + 1).trueLabel = 2
                            break
                                
                            #break
                #aspect is a single word
                else:
                    for ind, term in enumerate(nodes):
                        if term == aspect and tree.get(ind).trueLabel == 0:
                            tree.get(ind).trueLabel = 1
        
        #add pos tag in the tree
        for ind, term in enumerate(nodes):
            if ind > 0:

                tree.get(ind).pos = sequence[ind - 1][1]
            
            '''
            for term in nodes:
                ind = nodes.index(term)
                if term != None:
                    tree.get(ind).word = term.lower()
                    if term.lower() in opinion_words:
                        tree.get(ind).trueLabel = 3

                if term in pos_dic.keys():

                    for key in aspect_BIO.keys():
                        key_word = key[:-1]
                        if term == key_word: 
                            key_count += 1
                            
                    if key_count < 1:
                        for key in aspect_BIO.keys():
                            key_word = key[:-1]
                            if term == key_word and aspect_BIO[key] == 'B':
                                tree.get(ind).trueLabel = 1
                                break
                
                            elif term == key_word and aspect_BIO[key] == 'I':
                                tree.get(ind).trueLabel = 2
                                break
                    '''
            
                    
        # add dependency edges between nodes
        for rel, deps in plist:
            par_ind, par_word = deps[0]
            kid_ind, kid_word = deps[1]
            tree.add_edge(par_ind, kid_ind, rel)

        tree_dict.append(tree)  
        
        for node in tree.get_nodes():
            if node.word.lower() not in vocab:
                vocab.append(node.word.lower())
                
            node.ind = vocab.index(node.word.lower())
            
            for ind, rel in node.kids:
                if rel not in rel_list:
                    rel_list.append(rel)

        plist = []
        indice += 1
        
        
#word_embedding = gen.gen_word_embeddings(25, len(vocab))


print 'rels: ', len(rel_list)
print 'vocab: ', len(vocab)

cPickle.dump((vocab, rel_list, tree_dict), open("data_semEval/final_input_res_5class_new1", "wb"))
#cPickle.dump(word_embedding, open("initial_We_res", "wb"))


