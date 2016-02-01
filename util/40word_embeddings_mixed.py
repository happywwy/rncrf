# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:10:53 2015

@author: wangwenya
"""

import numpy as np
import cPickle

dic_file = open("../w2v_output_size50.txt", "r")
dic = dic_file.readlines()

dictionary = {}

for line in dic:
    word_vector = line.split(",")
    word = word_vector[0]
    
    vector_list = []
    for element in word_vector[1:len(word_vector)-1]:
        vector_list.append(element)
        
    vector = np.asarray(vector_list)
    dictionary[word] = vector
    
final_input = cPickle.load(open("data_semEval/final_input_res", "rb"))
vocab = final_input[0]

word_embedding = np.zeros((100, len(vocab)))

for ind, word in enumerate(vocab):
    if word in dictionary.keys():
        vec = dictionary[word]
        row = 0
        for num in vec:
            word_embedding[row][ind] = float(num)
            row += 1
        for r in range(50, 100):
            word_embedding[r][ind] = 2 * np.random.rand() - 1

    else:
        for i in range(100):
            word_embedding[i][ind] = 2 * np.random.rand() - 1

cPickle.dump(word_embedding, open("data_semEval/word_embeddings_mixed_res", "wb"))
