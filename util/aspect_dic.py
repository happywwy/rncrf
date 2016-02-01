# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 18:44:52 2015

@author: happywwy1991
"""
import cPickle

aspect_file = open("aspectTerm_laptop_BIO", "r")
dic = open("aspectTerm_laptop_dic", "wb")
aspect_dic = {}

aspects = aspect_file.read().splitlines()

for line in aspects:
    if line.strip() and "NIL" not in line:
        aspect_terms = line.split(',')
        
        for term in aspect_terms:
            if term in aspect_dic.keys():
                aspect_dic[term] += 1
            else:
                aspect_dic[term] = 1
                
#print aspect_dic

cPickle.dump(aspect_dic, dic)

aspect_file.close()
dic.close()
        