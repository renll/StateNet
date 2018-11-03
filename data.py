# -*- coding: utf-8 -*-

import os
import copy
import time
import pickle
import argparse
import json
import math
import numpy as np

import dataset_walker


cur_dir = os.path.dirname(os.path.abspath(__file__))
ontology_path = os.path.join(cur_dir, "config/ontology_dstc2.json")
vocab_path = os.path.join(cur_dir, 'vocab.dict')

# TODO !!! For consistency, this should be the ONLY place for loading ontology and vocab, and modifying them.
#      !!! Other files should import these data from here.
vocab = pickle.load(open(vocab_path,'rb'))
ontologyDict = json.load(open(ontology_path, 'r'))
for key in ontologyDict[u'informable']:
    ontologyDict[u'informable'][key].append('dontcare')

# TODO
#max_tag_index = 5
#for i in xrange(1, max_tag_index+1):
#    ontologyDict['informable']['food'].append('#food%d#'%i)
#    ontologyDict['informable']['name'].append('#name%d#'%i)


# ##################################

# TODO replace un-informable values with tags, for example:
# a machine act is "inform(addr='alibaba qijian dian')", then it is replaced as "inform(addr=<addr>)"
# default to [] to disable any such replacement
#replace_un_informable_slots = ['phone', 'postcode', 'addr']
replace_un_informable_slots = []

label_slot_order = ['food', 'pricerange', 'name', 'area']

def label2vec(labelDict, method, reqList):
    '''
    Parameters:
        1. goal
        2. method
        3. requests
    Return Value:
        1. resIdx
    '''
    resIdx = list()

    for slot in label_slot_order:
        if slot in labelDict and labelDict[slot] in ontologyDict['informable'][slot]:
            resIdx.append(ontologyDict['informable'][slot].index(labelDict[slot]))
        else:
            # the max index is for the special value: "none"
            resIdx.append(len(ontologyDict['informable'][slot]))

    resIdx.append(ontologyDict['method'].index(method))

    reqVec = [0.0] * len(ontologyDict['requestable'])
    for req in reqList:
        reqVec[ontologyDict['requestable'].index(req)] = 1
    resIdx.append(reqVec)

    return resIdx

def genTurnData_nbest(turn, labelJson):
    turnData = dict()

    # process user_input : exp scores
    user_input = turn["input"]["live"]["asr-hyps"]
    for asr_pair in user_input:
        asr_pair['score'] = math.exp(float(asr_pair['score']))

    # process machine_output : replace un-informable value with tags
    machine_output = turn["output"]["dialog-acts"]
    for slot in replace_un_informable_slots :
        for act in machine_output:
            for pair in act["slots"]:
                if len(pair) >= 2 and pair[0] == slot:
                    pair[1] = '<%s>' % slot

    # generate labelIdx
    labelIdx = label2vec(labelJson['goal-labels'], labelJson['method-label'], labelJson['requested-slots'])

    turnData["user_input"] = user_input
    turnData["machine_output"] = machine_output
    turnData["labelIdx"] = labelIdx
    return turnData

# ##################################
def tagTurnData(turnData, ontology):
    """将一个turn的数据进行tag替换"""
    tagged_turnData = copy.deepcopy(turnData)
    tag_dict = {}
    for slot in ["food", "name"]:
        val_ind = 1
        for slot_val in ontology["informable"][slot]:
            if slot_val.startswith("#%s"%slot):
                continue
            cur_tag = "#%s%d#" % (slot, val_ind)
            replace_flag = False

            # process user_input
            for i in xrange(len(tagged_turnData["user_input"])):
                sentence = tagged_turnData["user_input"][i]['asr-hyp']
                tag_sentence = sentence.replace(slot_val, cur_tag)
                if tag_sentence != sentence:
                    tagged_turnData["user_input"][i]['asr-hyp'] = tag_sentence
                    tag_dict[cur_tag] = slot_val
                    replace_flag = True

            # process machine_output
            for act in tagged_turnData["machine_output"]:
                for pair in act["slots"]:
                    if len(pair) >= 2 and pair[0] == slot and pair[1] == slot_val:
                        pair[1] = cur_tag
                        tag_dict[cur_tag] = slot_val
                        replace_flag = True

            if replace_flag:
                val_ind += 1
            if val_ind > max_tag_index:
                break

        # process labelIdx
        val_ind_dict = {ontology["informable"][slot].index(v):ontology["informable"][slot].index(k)
                        for k, v in tag_dict.items() if k.startswith("#%s"%slot)}
        labelIdx_ind = label_slot_order.index(slot)
        labelIdx = tagged_turnData["labelIdx"][labelIdx_ind]
        if labelIdx in val_ind_dict:
            tagged_turnData["labelIdx"][labelIdx_ind] = val_ind_dict[labelIdx]

        # add tag_dict to tagged_turnData
        tagged_turnData["tag_dict"] = tag_dict


    return tagged_turnData

def genTurnData_nbest_tagged(turn, labelJson):
    turnData = genTurnData_nbest(turn, labelJson)
    turnData = tagTurnData(turnData, ontologyDict)
    return turnData

# ##################################
def main():
    parser = argparse.ArgumentParser(description='Simple hand-crafted dialog state tracker baseline.')
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True,
                        help='The dataset to analyze')
    parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH',
                        help='Will look for corpus in <destroot>/<dataset>/...')
    parser.add_argument('--output_type',dest='output_type',action='store',default='nbest',
                        help='the type of output json')
    args = parser.parse_args()
    dataset = dataset_walker.dataset_walker(args.dataset, dataroot=args.dataroot, labels=True)

    def gen_data(func_genTurnData):
        data = []
        for call in dataset:
            fileData = dict()
            fileData["session-id"] = call.log["session-id"]
            fileData["turns"] = list()
            #print {"session-id":call.log["session-id"]}
            for turn, labelJson in call:
                turnData = func_genTurnData(turn, labelJson)
                fileData["turns"].append(turnData)
            data.append(fileData)
        return data

    # different output type
    if args.output_type == 'nbest':
        res_data = gen_data(genTurnData_nbest)
    elif args.output_type == 'nbest_tagged':
        res_data1 = gen_data(genTurnData_nbest)
        res_data2 = gen_data(genTurnData_nbest_tagged)
        res_data = res_data1 + res_data2

    # write to json file
    file_prefix = args.dataset.split('_')[-1]
    res_file = "%s_%s.json" % (file_prefix, args.output_type)
    with open(res_file, "w") as fw:
        fw.write(json.dumps(res_data, indent=2))


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print 'time: ', end_time - start_time, 's'
