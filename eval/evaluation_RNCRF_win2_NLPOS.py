# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:54:54 2015

@author: wangwenya
"""
import numpy as np
from rnn.crf_propagation import *
import cPickle
from sklearn import metrics
import pycrfsuite
#for ordered dictionary
from collections import OrderedDict

def pos2vec(pos):
    
    pos_list = ['PUNCT', 'SYM', 'CONJ', 'NUM', 'DET', 'ADV', 'X', 'ADP', 'ADJ', 'VERB', \
    'NOUN', 'PROPN', 'PART', 'PRON', 'INTJ']
            
    ind = pos_list.index(pos)
    vec = np.zeros(15)
    vec[ind] = 1
            
    return vec


#convert word to its namelist feature
def name2vec(sent, i, name_term, name_word):
    
    word = sent[i]
    name_vec = [0., 0.]
    if word != None:
        for term in name_term:
            if word == term:
                name_vec[0] = 1.
            elif i == 0 and len(sent) > 1 and sent[i + 1] != None:
                if word + ' ' + sent[i + 1] in term:
                    name_vec[0] = 1.
            elif i == len(sent) - 1 and len(sent) > 1 and sent[i - 1] != None:
                if sent[i - 1] + ' ' + word in term:
                    name_vec[0] = 1.
            elif i > 0 and i < len(sent) - 1:
                if (sent[i + 1] != None and word +' '+ sent[i + 1] in term) \
                    or (sent[i - 1] != None and sent[i - 1] +' '+ word in term):
                    
                    name_vec[0] = 1.
                
        if word in name_word:
            name_vec[1] = 1.
        
    return name_vec

    
#def word2features(sent, h_input, pos_mat, i):
def word2features(sent, h_input, pos_mat, name_term, name_word, i, d):
    
    #for ordered dictionary
    word_features = OrderedDict()    
    
    #word_features = {}
    word_features['bias'] = 1.

    #if it is punctuation
    if sent[i] == None:
        word_features['punkt'] = 1.
    
    else:    
        for n in range(d):
            word_features['worde=%d' % n] = h_input[i,n]
       
    #add pos features
    for n in range(15):
        word_features['pos=%d' % n] = pos_mat[i, n]
  
    
    #add namelist features
    name_vec = name2vec(sent, i, name_term, name_word)
    word_features['namelist1'] = name_vec[0]
    word_features['namelist2'] = name_vec[1]
    
    if i > 0 and sent[i - 1] == None:
        word_features['-1punkt'] = 1.  
        
    elif i > 0:
        for n in range(d):
            word_features['-1worde=%d' %n] = h_input[i - 1, n]
        
        #add pos features
        for n in range(15):
            word_features['-1pos=%d' % n] = pos_mat[i - 1, n]
        
        #add namelist features
        name_vec = name2vec(sent, i - 1, name_term, name_word)
        word_features['-1namelist1'] = name_vec[0]
        word_features['-1namelist2'] = name_vec[1]
        
        
    else:
        word_features['BOS'] = 1.
     
    if i > 1 and sent[i - 2] == None:
        word_features['-2punkt'] = 1.
        
    elif i > 1:
        for n in range(d):
            word_features['-2worde=%d' %n] = h_input[i - 2, n]
            
        #add pos features
        for n in range(15):
            word_features['-2pos=%d' % n] = pos_mat[i - 2, n]
        #add namelist features
        name_vec = name2vec(sent, i - 2, name_term, name_word)
        word_features['-2namelist1'] = name_vec[0]
        word_features['-2namelist2'] = name_vec[1]
            
    elif i == 1:
        word_features['BOS+1'] = 1.
    
    
    if i < len(sent) - 1 and sent[i + 1] == None:
        word_features['+1punkt'] = 1.
        
    elif i < len(sent) - 1:
        for n in range(d):
            word_features['+1worde=%d' %n] = h_input[i + 1, n]
        
        #add pos features
        for n in range(15):
            word_features['+1pos=%d' % n] = pos_mat[i + 1, n]
       
        #add namelist features
        name_vec = name2vec(sent, i + 1, name_term, name_word)
        word_features['+1namelist1'] = name_vec[0]
        word_features['+1namelist2'] = name_vec[1]
        
        
    else:
        word_features['EOS'] = 1.
   
    if i < len(sent) - 2 and sent[i + 2] == None:
        word_features['+2punkt'] = 1.
    
    elif i < len(sent) - 2:
        for n in range(d):
            word_features['+2worde=%d' %n] = h_input[i + 2, n]
        #add pos features
        for n in range(15):
            word_features['+2pos=%d' % n] = pos_mat[i + 2, n]
       
        #add namelist features
        name_vec = name2vec(sent, i + 2, name_term, name_word)
        word_features['+2namelist1'] = name_vec[0]
        word_features['+2namelist2'] = name_vec[1]
    elif i == len(sent) - 2:
        word_features['EOS-1'] = 1.
    
    
        
    return word_features


def sent2features(sent, h_input, pos_mat, name_term, name_word, d):
    return pycrfsuite.ItemSequence([word2features(sent, h_input, pos_mat, name_term, name_word, i, d) for i in range(len(sent))])
   
    
def evaluate(data_split, model_file, d, c, mixed = False):
    #output labels
    compare = open('out_label_res_namepos', 'w')
    tagger = pycrfsuite.Tagger()
    tagger.open('012crf.model')
    
    #find the 2 namelists
    f_term = open('util/data_semEval/namelist_term', 'rb')
    f_word = open('util/data_semEval/namelist_word', 'rb')
    
    name_term = cPickle.load(f_term)
    name_word = cPickle.load(f_word)
    
    f_term.close()
    f_word.close()

    dic_file = open('/home/wenya/Word2Vec_python_code_data/data/w2v_yelp300_10.txt', 'r')
    dic = dic_file.readlines()

    dictionary = {}

    for line in dic:
        word_vector = line.split(",")
        #word = word_vector[0]
        word = ','.join(word_vector[:len(word_vector) - d - 1])
    
        vector_list = []
        #for element in word_vector[1:len(word_vector)-1]:
        for element in word_vector[len(word_vector) - d - 1:len(word_vector) - 1]:
            vector_list.append(float(element))
        
        vector = np.asarray(vector_list)
        dictionary[word] = vector
    
    
    rel_list, tree_dict = \
        cPickle.load(open(data_split, 'rb'))
        
    test_trees = tree_dict
    [rel_dict, Wv, b, We], vocab, rel_list = cPickle.load(open(model_file, 'rb'))
    
    bad_trees = []
    for ind, tree in enumerate(test_trees):
        if tree.get(0).is_word == 0:
            # print tree.get_words()
            bad_trees.append(ind)
            continue

    # print 'removed', len(bad_trees)
    for ind in bad_trees[::-1]:
        #test_trees.pop(ind)
        test_trees = np.delete(test_trees, ind)
      
    true = []
    predict = []  
    
    count = 0
    
    for ind, tree in enumerate(test_trees):
        nodes = tree.get_nodes()
        sent = []
        h_input = np.zeros((len(tree.nodes) - 1, d))
        y_label = np.zeros((len(tree.nodes) - 1,), dtype = int)
       
        #add pos matrix
        pos_mat = np.zeros((len(tree.nodes) - 1, 15))
        
        for index, node in enumerate(nodes):
            if node.word.lower() in vocab:
                node.vec = We[:, node.ind].reshape( (d, 1) )
            elif node.word.lower() in dictionary.keys():
                if mixed:
                    node.vec = (dictionary[node.word.lower()].append(2 * np.random.rand(50) - 1)).reshape( (d, 1) )
                else:
                    node.vec = dictionary[node.word.lower()].reshape(d, 1)
            else:
                node.vec = np.random.rand(d,1)
                count += 1
            
        forward_prop([rel_dict, Wv, b, We], tree, d, c, labels=False)
        
        for index, node in enumerate(tree.nodes): 
            
            if index != 0:
                
                if tree.get(index).is_word == 0:
                    y_label[index - 1] = 0
                    sent.append(None)
                    
                    for i in range(d):
                        h_input[index - 1][i] = 0
                else:
                    y_label[index - 1] = node.trueLabel
                    sent.append(node.word)
                    
                    for i in range(d):
                        h_input[index - 1][i] = node.p[i]
                #sent.append(node.word)
                
                #get pos vector
                pos = node.pos
                pos_vec = pos2vec(pos)
                
                for i in range(15):
                    pos_mat[index - 1, i] = pos_vec[i]

        crf_sent_features = sent2features(sent, h_input, pos_mat, name_term, name_word, d)
        for item in y_label:
            true.append(str(item))
           
        compare.write(''.join(str(item) for item in y_label))
        compare.write('\n')
        
        #predict           
        prediction = tagger.tag(crf_sent_features)
        for label in prediction:
            predict.append(label)
        
        compare.write(''.join(item for item in prediction))
        compare.write('\n')
        

    compare.close()
    print (metrics.classification_report(true, predict))
    print "Confusion matrix from sklearn: \n", metrics.confusion_matrix(true,predict)
    
    print metrics.precision_recall_fscore_support(true, predict, average = 'macro')
    
    import time 
    timeStr=time.strftime("%m%d-%H%M%S")
    
    tagger.dump('stdout'+timeStr)
    
    print count

