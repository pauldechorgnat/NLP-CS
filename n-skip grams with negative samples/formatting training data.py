# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:31:23 2018

@author: Paul
"""

path = 'train_corpus.txt'

text = []
full_text = ""
with open(path, 'r', encoding = 'cp1252') as input_file:
    for l in input_file:
        full_text+=l.replace("\n", "")
input_file.close()

text = full_text.split(".")
with open("formatted_train_corpus.txt", 'w', encoding = 'utf-8') as output_file:
    for sentence in text:
        output_file.write(sentence+"\n")
output_file.close()
        
    