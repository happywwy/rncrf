# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:14:30 2015

@author: wangwenya
"""

import xml.etree.ElementTree as ET

tree = ET.parse('raw_data/Laptops_Test.xml')
root = tree.getroot()

parsed_sentence = []
output = open('util/data_semEval/sentence_laptest.txt', 'w')

for child in root:
    output.write(child[0].text.encode("utf-8"))
    output.write('\n')
    
"""
    
for child in root:
    for kid in child[0]:
        output.write(kid[0].text)
        output.write('\n')
"""    
output.close()