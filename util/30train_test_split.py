# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:11:00 2015

@author: wangwenya
"""

import sys, cPickle, random, os
from numpy import *
from sklearn.cross_validation import train_test_split

"""
#from training set
vocab, rel_list, tree_dic_train = cPickle.load(open("final_input", "rb"))

#from test set
test = cPickle.load(open("final_input_test", "rb"))
tree_dic_test = test[1]

#combine train and test trees
tree_dic = {}
tree_dic['train'] = tree_dic_train
tree_dic['test'] = tree_dic_test

cPickle.dump((vocab, rel_list, tree_dic), open("train_test_split", "wb"))
"""
#for new data from customer review
vocab, rel_list, tree_dic = cPickle.load(open("data_semEval/final_input_res", "rb"))

train_dic, test_dic = train_test_split(tree_dic, test_size = 0.15)
tree = {}
tree['train'] = train_dic
tree['test'] = test_dic

cPickle.dump((vocab, rel_list, tree), open("data_semEval/train_test_split_res", "wb"))
cPickle.dump((vocab, rel_list, train_dic), open("data_semEval/final_training_res", "wb"))