# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:30:13 2015

@author: wenya
"""

import cPickle

f_term = open('aspectTerm_laptop_dic', 'rb')
f_word = open('aspectWord_laptop_dic', 'rb')

term_dic = cPickle.load(f_term)
word_dic = cPickle.load(f_word)
namelist_term = []
namelist_word = []

for key in term_dic.keys():
    if term_dic[key] >= 2:
        namelist_term.append(key.lower())
        
for key in word_dic.keys():
    if word_dic[key] >=0.1:
        namelist_word.append(key.lower())
        
term = open('namelist_term_laptop', 'wb')
word = open('namelist_word_laptop', 'wb')

#print namelist_word

cPickle.dump(namelist_term, term)
cPickle.dump(namelist_word, word)

f_term.close()
f_word.close()
term.close()
word.close()