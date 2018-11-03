# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))

EMBEDDING_DIM =300
vocab_version = 'vN3'

pretrained_file ='paragram_300_sl999.txt'#
vocab_dict_file = 'vocab_matNN.dict'
embedding_matrix_file = 'embed_%s.npy' % vocab_version
vocab_embed_txt = 'vocab2embed_%s.txt' % vocab_version


# load vocab_dict
vocab_dict = pickle.load(open(os.path.join(cur_dir, vocab_dict_file)))
#print vocab_dict



# load pretrained embeddings
embeddings_index = {}

embedding_matrix = np.zeros((len(vocab_dict), EMBEDDING_DIM))
#print vocab_dict.keys() 
   
m=0   
with open(os.path.join(cur_dir, pretrained_file)) as f:
    for line in f:
        values = line.split()
        word = values[0]
        if word in vocab_dict.keys():
            coefs = np.asarray(values[1:])
            embedding_matrix[vocab_dict.get(word)]=coefs
            m+=1
print 'Found %s word vectors.' % m
print len(coefs)
embedding_matrix[1] = np.asarray([0.0]*EMBEDDING_DIM)
embedding_matrix[0] = np.asarray([0.0]*EMBEDDING_DIM)

save_path = os.path.join(cur_dir, 'vocab_set', embedding_matrix_file)
np.save(save_path, embedding_matrix)
print 'Save embedding_matrix to:', save_path



# generate vocab2embed_list
vocab2embed_list = [None] * len(vocab_dict)
for word, i in vocab_dict.items():
    vocab2embed_list[i] = [word] + [str(num) for num in embedding_matrix[i].tolist()]
file_string = '\n'.join([' '.join(item) for item in vocab2embed_list])

save_path = os.path.join(cur_dir, 'vocab_set', vocab_embed_txt)
with open(save_path, 'w') as f:
    f.write(file_string)
print 'Save vocab_embed_txt to:', save_path

