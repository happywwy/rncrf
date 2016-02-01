# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:28:58 2015

@author: wangwenya
"""

import xml.etree.ElementTree as ET

tree = ET.parse('raw_data/Laptops_Test.xml')
#tree = ET.parse('raw_data/Restaurants_Test.xml')
root = tree.getroot()

aspect_Term = []
output = open('util/data_semEval/aspectTerm_laptest_BIO', 'w')
#output = open('raw_data/aspectTerm_restest_BIO', 'w')
counter = 0
merge_Start = 0

for child in root:
    #for restaurant domain
    #if len(child) > 2:
    #for laptop domain
    if len(child) > 1:
        if len(child[1]) > 1:
            merge_Start = counter
            for kid in child[1]:
                #aspect_Term.append(kid.attrib["term"] + ',' + kid.attrib["polarity"] + ' ')
                aspect_Term.append(kid.attrib["term"])
                counter = counter + 1
                
            aspect_Term[merge_Start:counter] = [','.join(aspect_Term[merge_Start:counter])]
            counter = merge_Start + 1
            
        else:
            aspect_Term.append(child[1][0].attrib["term"])
            counter = counter + 1
            
    else:
        aspect_Term.append("NIL")
        counter = counter + 1

'''
for child in root:
    for kid in child[0]:
        if len(kid) > 1:
            if len(kid[1]) > 1:
                merge_Start = counter
                num = 0
                for element in kid[1]:
                    if element.attrib["target"] != "NULL":
                        aspect_Term.append(element.attrib["target"])
                        #aspect_Term.append(element.attrib["target"] + ',' + element.attrib["polarity"] + ';')
                        counter = counter + 1
                    else:
                        num += 1
                if num == len(kid[1]):
                    aspect_Term.append("NIL")
                    counter += 1
                else:
                    aspect_Term[merge_Start:counter] = [','.join(aspect_Term[merge_Start:counter])]
                    counter = merge_Start + 1
            
            elif kid[1][0].attrib["target"] != "NULL":
                aspect_Term.append(kid[1][0].attrib["target"])
                #aspect_Term.append(kid[1][0].attrib["target"] + ',' + kid[1][0].attrib["polarity"])
                counter = counter + 1
            
            else:
                aspect_Term.append("NIL")
                counter = counter + 1
                
        else:
            aspect_Term.append("NIL")
            counter += 1
'''     
for term in aspect_Term:
    output.write(term.encode('utf-8'))
    output.write('\n')
    
output.close()
            
    #print child[1][0].attrib["term"]
