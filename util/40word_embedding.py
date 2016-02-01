# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:50:17 2015

@author: happywwy1991
"""

import numpy as np
import cPickle

#dic_file = open("../w2v_yelp400_10.txt", "r")
dic_file = open("/home/wenya/Word2Vec_python_code_data/data/w2v_amazon375_5.txt", "r")
#dic_file = open("../hlbl_50.txt", "r")
dic = dic_file.readlines()

dictionary = {}

for line in dic:
    word_vector = line.split(",")
    #word_vector = line.split()
    i = 1
    while '.' not in word_vector[i] or '..' in word_vector[i] or word_vector[i] == '.':
        i += 1
        
    word = ','.join(word_vector[:i])
    
    vector_list = []
    for element in word_vector[i:len(word_vector)-1]:
        vector_list.append(element)
        
    vector = np.asarray(vector_list)
    dictionary[word] = vector
    

final_input = cPickle.load(open("data_semEval/final_input_laptop_5class_lower", "rb"))
vocab = final_input[0]

word_embedding = np.zeros((375, len(vocab)))

count = 0

for ind, word in enumerate(vocab):
    if word in dictionary.keys():
        vec = dictionary[word]
        row = 0
        for num in vec:
            word_embedding[row][ind] = float(num)
            row += 1
        count += 1
    else:
        print word,
        for i in range(375):
            word_embedding[i][ind] = 2 * np.random.rand() - 1
    
print len(vocab)
print count
#print word_embedding


cPickle.dump(word_embedding, open("data_semEval/word_embeddings375_5class_laptop", "wb"))
