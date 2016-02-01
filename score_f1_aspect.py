# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:30:53 2016

@author: wenya
"""

f = open('out_label_laptop_namepos', 'r')

lines = f.read().splitlines()
correct = 0
predicted = 0
relevant = 0

i=0
j=0
pairs = []
while i < len(lines):
    true_seq = lines[i]
    predict = lines[i+1]
    
    for num in range(len(true_seq)):
        if true_seq[num] == '1':
            if num < len(true_seq) - 1:
                if true_seq[num + 1] == '0' or true_seq[num + 1] == '1':
                    if predict[num] == '1':
                        correct += 1
                        #predicted += 1
                        relevant += 1
                    else:
                        relevant += 1
                
                else:
                    if predict[num] == '1':
                        for j in range(num + 1, len(true_seq)):
                            if true_seq[j] == '2':
                                if predict[j] == '2' and j < len(predict) - 1:
                                    continue
                                elif predict[j] == '2' and j == len(predict) - 1:
                                    correct += 1
                                    relevant += 1
                                    
                                else:
                                    relevant += 1
                                    break
                                
                            else:
                                correct += 1
                                #predicted += 1
                                relevant += 1
                                break

                            
                    else:
                        relevant += 1
                        
            else:
                if predict[num] == '1':
                    correct += 1
                    #predicted += 1
                    relevant += 1
                else:
                    relevant += 1
                    
                        
    for num in range(len(predict)):
        if predict[num] == '1':
            if num < len(predict) - 1:
                if predict[num + 1] == '0' or predict[num + 1] == '1':
                    predicted += 1
                else:
                    for j in range(num + 1, len(predict)):
                        if predict[j] != 2:
                            predicted += 1
                            break
                        
            else:
                predicted += 1
                    
                    
    i += 2
            
precision = float(correct) / predicted
recall = float(correct) / relevant
f1 = 2 * precision * recall / (precision + recall)

print precision
print recall
print f1                   
    