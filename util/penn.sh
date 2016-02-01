#!/bin/bash

#
# Parse with Stanford Parser (parse trees and dependencies)
#

#MODEL=edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz
MODEL=englishPCFG.ser.gz

F=./data_semEval/processed_laptest

G=./data_semEval/penn_tree_laptest
    
java -mx2048m -cp "*" edu.stanford.nlp.parser.lexparser.LexicalizedParser -nthreads 2 -sentences newline \
-retainTmpSubcategories -outputFormat "penn" -outputFormatOptions "basicDependencies" ${MODEL} ${F} > ${G}

