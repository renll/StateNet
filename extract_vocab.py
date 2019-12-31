# -*- coding: utf-8 -*-

import pickle
import time
import os
import json
#from collections import OrderedDict

import dataset_walker

# ontology所在位置
cur_dir = os.path.dirname(os.path.abspath(__file__))
ontology_path = os.path.join(cur_dir, 'config/ontology_dstc2.json')
ontology = json.load(open(ontology_path, 'r'))

dataset_name = 'dstc2_train'
dataroot = 'dstc2_traindev/data'

start_time = time.time()


# vocab to be generated
vocab_dict = {}
# 词表中每个词出现的最小词频
oov_threshold = 20
# 出现过的所有词
word_dict = {}

# pre-set some words
vocab_dict["<oov>"]     = len(vocab_dict)
vocab_dict["</s>"]      = len(vocab_dict)
vocab_dict["#turn#"]    = len(vocab_dict)

# TODO
vocab_dict["#food#"]   = len(vocab_dict)
# vocab_dict["#food2#"]   = len(vocab_dict)
# vocab_dict["#food3#"]   = len(vocab_dict)
# vocab_dict["#food4#"]   = len(vocab_dict)
# vocab_dict["#food5#"]   = len(vocab_dict)
vocab_dict["#name#"]   = len(vocab_dict)
vocab_dict["#slot#"]   = len(vocab_dict)
# vocab_dict["#name2#"]   = len(vocab_dict)
# vocab_dict["#name3#"]   = len(vocab_dict)
# vocab_dict["#name4#"]   = len(vocab_dict)
# vocab_dict["#name5#"]   = len(vocab_dict)

# TODO
# vocab_dict["<addr>"]     = len(vocab_dict)
# vocab_dict["<phone>"]    = len(vocab_dict)
# vocab_dict["<postcode>"] = len(vocab_dict)

def add_word(word):
    word=word.encode('utf-8')
    """向word_dict中添加一个word"""
    word_dict[word] = word_dict.get(word, 0) + 1

def add_words(words):
    """向word_dict中添加若干个words"""
    word_list = words.split()
    # add 1-gram word
    for word in word_list:
        add_word(word)
    # TODO add 2-gram word
    #for word in [' '.join(word_list[i:i+2]) for i in xrange(len(word_list)-1)]:
    #    add_word(word)
    # TODO add 3-gram word
    #for word in [' '.join(word_list[i:i+3]) for i in xrange(len(word_list)-2)]:
    #    add_word(word)


# include all ontology values into vocab
# TODO
add_words("none")
add_words("dontcare")
   
for i in range(59):
    for key in ontology:
        add_words(key)
        if key in ["requestable", "method"]:
            for val in ontology[key]:
                add_words(val)
        elif key == "informable":
            for slot in ["area", "pricerange"]:
                add_words(slot)
                for val in ontology[key][slot]:
                    add_words(val)
            # TODO
            for slot in ["food"]:
                add_words(slot)
                for val in ontology[key][slot]:
                    add_words(val)
            # TODO
            for slot in ["name"]:
                add_words(slot)
                for val in ontology[key][slot]:
                    add_words(val)


# include asr words and slu words appeared in data set
dataset = dataset_walker.dataset_walker(dataset_name, dataroot=dataroot, labels=True)
add_words("asr")
add_words("slots")
add_words("act")
for call in dataset:
    for turn, labelJson in call:
        asrs = turn["input"]["live"]["asr-hyps"]

        # 1best
        add_words(asrs[0]["asr-hyp"])

        # 2best - nbest
        # TODO
        for asr in asrs[1:]:
            add_words(asr["asr-hyp"])

        # dialog acts
        machine_act_words = []
        for act_item in turn["output"]["dialog-acts"]:
            if "act" in act_item:
                machine_act_words.append(act_item["act"])
            if "slots" in act_item:
                for item in act_item["slots"]:
                    for item_val in item:
                        machine_act_words.append(item_val)
        machine_act = ' '.join(machine_act_words)
        add_words(machine_act)


# save vocab to file
# TODO modify file name if needed
with open('vocab_matNN.dict', 'wb') as f:
    for word, freq in word_dict.items():
        if freq >= oov_threshold:
            vocab_dict[word] = len(vocab_dict)
    pickle.dump(vocab_dict, f)


end_time = time.time()
print "vocab size:", len(vocab_dict)
#print vocab_dict
print "cost time: ", end_time-start_time, 's'
