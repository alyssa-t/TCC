import nltk  
import numpy as np  
import random  
import string

import bs4 as bs  
import urllib.request  
import re 

f = open("en-captions.txt","r")
fTxt = f.read()
f.close()

corpus = nltk.sent_tokenize(fTxt)

for i in range(len(corpus )):
    corpus [i] = corpus [i].lower()
    corpus [i] = re.sub(r'\W',' ',corpus [i])
    corpus [i] = re.sub(r'\s+',' ',corpus [i])


print(len(corpus))

wordfreq = {}
for sentence in corpus:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1

f = open("test.txt","w")
for k, v in sorted(wordfreq.items(), reverse=True, key=lambda item: item[1]) : 
    f.write(str(k)+ " "+ str(v)+"\n")

f.close()