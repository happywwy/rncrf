#!/bin/bash

#
# Parse with Stanford Parser (parse trees and dependencies)
#


#MODEL=edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz
#MODEL=englishPCFG.ser.gz

F=./data_semEval/penn_tree_laptest

G=./data_semEval/parse_conll_laptest
    
java -mx2048m -cp "*" edu.stanford.nlp.trees.EnglishGrammaticalStructure -nthreads 2 \
-retainTmpSubcategories -treeFile ${F} -basic -conllx >${G}


