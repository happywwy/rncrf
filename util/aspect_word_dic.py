# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:06:05 2015

@author: happywwy1991
"""

import cPickle

aspect_file = open("aspectTerm_laptop_BIO", "r")
sent_file = open("processed_laptop", "r")
dic = open("aspectWord_laptop_dic", "wb")
aspect_dic = {}

aspects = aspect_file.read().splitlines()

for line in aspects:
    if line.strip() and "NIL" not in line:
        aspect_terms = line.split(',')
        
        for term in aspect_terms:
            words = term.split()
            
            for word in words:
                if word != '':
                    if "(" in word:
                        word = word.replace('(', '')
                    if ")" in word:
                        word = word.replace(")", '')
                        
                    if word in aspect_dic.keys():
                        aspect_dic[word] += 1
                    else:
                        aspect_dic[word] = 1
                
#print aspect_dic
                
#calculating the probability of being in aspect word
sentences = sent_file.read().splitlines()
all_dic = {}
prob_dic = {}

for key in aspect_dic.keys():
    all_dic[key] = 0

for key in aspect_dic.keys():
    for sentence in sentences:
        if ',' in sentence:
            sentence = sentence.replace(',', ' ')
        if "(" in sentence:
            sentence = sentence.replace('(', '')
        if ")" in sentence:
            sentence = sentence.replace(')', '')
            
        word_list = sentence.split()
        for word in word_list:
            if key == word or "'" + key + "'" == word or key + "'" == word or  "'" + key == word or key + "'s" == word:
                all_dic[key] += 1
                
#print all_dic
for key in aspect_dic.keys():
    if all_dic[key] != 0:
        prob_dic[key] = float(aspect_dic[key]) / all_dic[key]

#print prob_dic

             
cPickle.dump(prob_dic, dic)

aspect_file.close()
dic.close()
