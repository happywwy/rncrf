#!/bin/bash

#
# Parse with Stanford Parser (parse trees and dependencies)
#

#MODEL=edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz
MODEL=englishPCFG.ser.gz

F=tokens

G=raw_parseToken_res
    
java -mx2048m -cp "*" edu.stanford.nlp.parser.lexparser.LexicalizedParser -tokenized -escaper edu.stanford.nlp.process.PTBEscapingProcessor -sentences newline \
-retainTmpSubcategories -outputFormat "typedDependencies" -outputFormatOptions "basicDependencies" ${MODEL} ${F} > ${G}

