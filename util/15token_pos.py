# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:47:18 2015

@author: wenya
"""

"""
create (token, pos) pairs from dependency parser
"""

import cPickle

f = open('./data_semEval/parse_conll_laptest', 'r')
conll = f.read().splitlines()

tokens_all = []
tokens = []

'''
#PENN treebank
pos = []

for line in conll:
    
    if line.strip():
        line = line.split('\t')
        tokens.append([line[1], line[3]])
        if line[3] not in pos:
            pos.append(line[3])
    else:
        tokens_all.append(tokens)
        tokens = []
    
out = open('./data_semEval/token_pos_res', 'wb')
cPickle.dump(tokens_all, out)
out.close()
'''

#universal treebank
uni_pos = []
dic = {}
#conversion types
dic['PUNCT'] = [',', '``', "''", '-LRB-', '-RRB-', ':', 'LS']
dic['SYM'] = ['SYM']
dic['CONJ'] = ['CC']
dic['NUM'] = ['CD']
dic['DET'] = ['DT', 'PDT', 'PRP$', 'WDT', 'WP$']
dic['ADV'] = ['EX', 'RB', 'RBR', 'RBS', 'WRB']
dic['X'] = ['FW']
dic['ADP'] = ['IN']
dic['ADJ'] = ['JJ', 'JJR', 'JJS']
dic['VERB'] = ['MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
dic['NOUN'] = ['NN', 'NNS']
dic['PROPN'] = ['NNP', 'NNPS']
dic['PART'] = ['POS', 'RP', 'TO']
dic['PRON'] = ['PRP', 'WP']
dic['INTJ'] = ['UH']


for line in conll:
    if line.strip():
        line = line.split('\t')
        
        for key in dic.keys():
            if line[3] in dic[key]:
                tokens.append([line[1], key])
    
    else:
        tokens_all.append(tokens)
        tokens = []

out = open('./data_semEval/uni_token_pos_laptest', 'wb')
cPickle.dump(tokens_all, out)
out.close()

f.close()


#print pos