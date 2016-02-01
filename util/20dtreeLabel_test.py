# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:28:35 2015

@author: wangwenya
"""

"""
create tree structures from raw parses for testing sentences
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
from nltk.tokenize import word_tokenize
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

f_test = open('data_semEval/raw_parses_restest_new', 'r')
#sentence_file = open('data_semEval/parsedSentence_restest800.txt', 'r')

f_token_pos = open('data_semEval/uni_token_pos_restest_new', 'rb')
token_pos = cPickle.load(f_token_pos)
indice = 0

data_test = f_test.readlines()
plist = []
tree_dict_test = []
rel_list = []

train_input = cPickle.load(open('data_semEval/final_input_res_5class_new', 'rb'))
vocab = train_input[0]

label_file = open('data_semEval/aspectTerm_restest_BIO', 'r')
label_sentence = open('data_semEval/addsenti_restest', 'r')

for line in data_test:
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
                nodes[ind] = word

        tree = dtree(nodes)
        '''
        sentence = sentence_file.readline().strip()
        terms = word_tokenize(sentence)
        pos_review = nltk.pos_tag(terms)
        '''
        
        
        sequence = token_pos[indice]
       
        
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
        aspect_term = label_file.readline().strip()
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
                            if term == op_list[0] and ind < len(nodes) - 1 and nodes[ind + 1] == op_list[1]:
                                tree.get(ind).trueLabel = 3
                                for i in range(len(op_list) - 1):
                                    if nodes[ind + i + 1] == op_list[i + 1]:
                                        tree.get(ind + i + 1).trueLabel = 4
                                        
                    elif len(op_list) == 1:
                        for ind, term in enumerate(nodes):
                            if term == op_list[0] and tree.get(ind).trueLabel == 0:
                                tree.get(ind).trueLabel = 3
        
        if aspect_term != 'NIL':
            #aspect_term += " "
            #line = aspect_term.split(' ')
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

                                if nodes[ind + i + 1] == aspect_list[i + 1]:
                                    tree.get(ind + i + 1).trueLabel = 2
                                
                            #break
                #aspect is a single word
                else:
                    for ind, term in enumerate(nodes):
                        if term == aspect and tree.get(ind).trueLabel == 0:
                            tree.get(ind).trueLabel = 1
        
        
        #add pos tag in the tree
        for ind, term in enumerate(nodes):
            if ind > 0:
                if ind > len(sequence):
                    print indice
                tree.get(ind).pos = sequence[ind - 1][1]
            
        for term in nodes:
            if term != None:
                ind = nodes.index(term)
                tree.get(ind).word = term.lower()

        # add dependency edges between nodes
        for rel, deps in plist:
            par_ind, par_word = deps[0]
            kid_ind, kid_word = deps[1]
            tree.add_edge(par_ind, kid_ind, rel)

        tree_dict_test.append(tree)  
        
        for node in tree.get_nodes():
            if node.word.lower() in vocab:
                
                node.ind = vocab.index(node.word.lower())
            
            for ind, rel in node.kids:
                if rel not in rel_list:
                    rel_list.append(rel)

        plist = []
        indice += 1


cPickle.dump((rel_list, tree_dict_test), open("data_semEval/final_input_restest_5class_new1", "wb"))



