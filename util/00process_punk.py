# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 00:43:53 2015

@author: happywwy1991
"""
"""
output file: "processed_res", "processed_restest"
task: preprocess raw sentences to delete punctuations, unified numbers

"""


sentence_file = open('data_semEval/sentence_laptop', 'r')
sentences = sentence_file.read().splitlines()

processed_file = open('data_semEval/processed_laptop', 'w')

punkt = ['/', '\\', '!', ';', '=', \
'+', '#', '$', '@', '%', '^', '&', '*', '?', '<', '>', '`']

keep = [':', ',', '-', '...', '--', '---']

num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
special = ["n't", "'s", "'ll", "'m", "'re", "'d", "'ve"]

for line in sentences:
    if any(pun in line for pun in punkt):
        for thing in [pun for pun in punkt if pun in line]:
            line = line.replace(thing, ' ')
    
    if '.' in line:
        line = line.replace('.', '')
        
    if '--' in line:
        line = line.replace('--', '-')
        
    
        
    tokens = line.split()
    for ind, word in enumerate(tokens):
        if any(n in word for n in num):
            tokens[ind] = 'NUM'
        '''
        if word == '-' or word == '--' or word == '---':
            tokens[ind] = ''
        '''    
    processed_file.write(' '.join(item for item in tokens))
    processed_file.write('\n')
    
sentence_file.close()
processed_file.close()
        
        