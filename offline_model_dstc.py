# -*- coding: utf-8 -*-

import os
import time
import json
import copy
import math
import numpy as np
import mxnet as mx
from offline_model import OfflineModel
from mod_lectrack import ModTracker
from mat_data import genTurnData_nbest, genTurnData_nbest_tagged,gen_resdata

import dataset_walker

# 脚本所在位置
cur_dir = os.path.dirname(os.path.abspath(__file__))

# TODO offline config
# nn_type can be ['lstm', 'blstm', 'lstmn', 'cnn', 'bowlstm']
modSel=1
if modSel==0:
# blstm config
    offline_config_dict = {
        'nn_type': 'lstmcnn',
        'model_dir': os.path.join(cur_dir, 'models_dstc_lstmcnnRes'),
        'train_json': os.path.join(cur_dir, 'train_nbest.json'),
        'dev_json': os.path.join(cur_dir, 'dev_nbest.json'),
        'test_json': os.path.join(cur_dir, 'test_nbest.json')
    }
elif modSel==1:
#lstm config
    offline_config_dict = {
       'nn_type': 'doublelstm',
       'model_dir': os.path.join(cur_dir, 'models_dstc_caplstmSCLT'),
       'train_json': os.path.join(cur_dir, 'train_nbest_tagged.json'),
       'dev_json': os.path.join(cur_dir, 'dev_nbest_tagged.json'),
       'test_json': os.path.join(cur_dir, 'test_nbest_tagged.json')
    }
else:
# mat config
    offline_config_dict = {
       'nn_type': 'reslstm',
       'model_dir': os.path.join(cur_dir, 'models_dstc_reslstmTest1'),
       'train_json': os.path.join(cur_dir, 'train_nbest_tagged.json'),
       'dev_json': os.path.join(cur_dir, 'dev_nbest_tagged.json'),
       'test_json': os.path.join(cur_dir, 'test_nbest_tagged.json')
    }


# 1best tagged lstm config
#offline_config_dict = {
#    'nn_type': 'lstm',
#    'model_dir': os.path.join(cur_dir, 'models_dstc2_tagged'),
#    'train_json': os.path.join(cur_dir, 'train_nbest_tagged.json'),
#    'dev_json': os.path.join(cur_dir, 'dev_nbest_tagged.json'),
#    'test_json': os.path.join(cur_dir, 'test_nbest_tagged.json')
#}

# cnn config
#offline_config_dict = {
#    'nn_type': 'cnn',
#    'model_dir': os.path.join(cur_dir, 'models_cnn'),
#    'train_json': os.path.join(cur_dir, 'train_nbest.json'),
#    'dev_json': os.path.join(cur_dir, 'dev_nbest.json'),
#    'test_json': os.path.join(cur_dir, 'test_nbest.json')
#}

# bowlstm config
#offline_config_dict = {
#    'nn_type': 'bowlstm',
#    'model_dir': os.path.join(cur_dir, 'models_bowlstm'),
#    'train_json': os.path.join(cur_dir, 'train_nbest.json'),
#    'dev_json': os.path.join(cur_dir, 'dev_nbest.json'),
#    'test_json': os.path.join(cur_dir, 'test_nbest.json')
#}

# cnnlstm config
#offline_config_dict = {
#    'nn_type': 'cnnlstm',
#    'model_dir': os.path.join(cur_dir, 'models_cnnlstm'),
#    'train_json': os.path.join(cur_dir, 'train_nbest.json'),
#    'dev_json': os.path.join(cur_dir, 'dev_nbest.json'),
#    'test_json': os.path.join(cur_dir, 'test_nbest.json')
#}

# cnncnnlstm config
# offline_config_dict = {
#     'nn_type': 'cnncnnlstm',
#     'model_dir': os.path.join(cur_dir, 'models_cnncnnlstm'),
#     'train_json': os.path.join(cur_dir, 'train_nbest.json'),
#     'dev_json': os.path.join(cur_dir, 'dev_nbest.json'),
#     'test_json': os.path.join(cur_dir, 'test_nbest.json')
# }

def train_dstc2(ini):
    #params_path = os.path.join(offline_config_dict['model_dir'], 'labelIdx-joint-testbest.params' )
    #if os.path.exists(params_path):
    #     os.remove(params_path)
    for t in range(1,2):
        np.random.seed(t)
        ctx = 'gpu'
        if ini==1:
            index=[0]
        else:
            index=[0,1,3]
        for i in [index]:
            tmp_model = OfflineModel(i, ini, ctx, offline_config_dict)
            tmp_model.offline_train(150)
            #tmp_model.offline_eval()

def del_none_val(turn_output):
    """delete all "none" values in turn_output. In-place operation"""
    if "none" in turn_output["requested-slots"]:
        del turn_output["requested-slots"]["none"]
    for _, vals in turn_output["goal-labels"].items():
        if "none" in vals:
            del vals["none"]

def tag_to_val(turn_output, tag_dict):
    goal_output = turn_output["goal-labels"]
    for slot in goal_output:
        for slotval in copy.deepcopy(goal_output[slot]):
            if slotval.startswith('#'):
                if slotval in tag_dict:
                    goal_output[slot][tag_dict[slotval]] = goal_output[slot].get(tag_dict[slotval], 0.0) + goal_output[slot][slotval]
                del goal_output[slot][slotval]


def gen_baseline_ground(dataset_name, dataroot):
    res_ground = {
        'dataset': dataset_name,
        'sessions': []
    }
    dataset = dataset_walker.dataset_walker(dataset_name, dataroot=dataroot, labels=True)
    for call in dataset:
        res_dialogue = dict()
        res_dialogue["session-id"] = call.log["session-id"]
        res_dialogue["turns"] = list()
        for turn, labelJson in call:
            turn_label = {
                "goal-labels": labelJson["goal-labels"],
                "method-label": labelJson["method-label"],
                "requested-slots": labelJson["requested-slots"]
            }
            res_dialogue["turns"].append(turn_label)
        res_ground["sessions"].append(res_dialogue)
    json.dump(res_ground, open('baseline_ground_%s.json'%dataset_name, 'wb'), indent=4)


def gen_baseline(dataset_name, dataroot, tagged=False):
    res = {
        'dataset': dataset_name,
        'sessions': []
    }
    dataset = dataset_walker.dataset_walker(dataset_name, dataroot=dataroot, labels=True)
    mod_config_dict = {
        'context_type': 'cpu',
        'nn_type': offline_config_dict["nn_type"],
        'model_dir':offline_config_dict["model_dir"]
    }
    if mod_config_dict['nn_type'] in ['doublelstm','reslstm','matlstm','cnnlstm', 'cnncnnlstm']:
        mod_config_dict['batch_size'] = 32

    mod_tracker = ModTracker(config_dict=mod_config_dict)
    start_time = time.time()

    # decide how to process data
    if mod_config_dict['nn_type'] in ['bowlstm']:
        level = 'turn'
        feature_type = 'bow'
    elif mod_config_dict['nn_type'] in ['reslstm','matlstm','cnnlstm']:
        level = 'turn'
        feature_type = 'bowbow'
    elif mod_config_dict['nn_type'] in ['doublelstm','cnncnnlstm']:
        level = 'turn'
        feature_type = 'sentbow'
    else:
        level = 'word'

    # process by word-level dialogue
    if level == 'word':
        for call in dataset:
            res_dialogue = dict()
            res_dialogue["session-id"] = call.log["session-id"]
            res_dialogue["turns"] = list()

            fileDatas = []
            tag_dicts = []

            fileData = {}
            fileData["turns"] = []
            for turn, labelJson in call:
                if tagged:
                    turnData = genTurnData_nbest_tagged(turn, labelJson)
                    tag_dicts.append(turnData["tag_dict"])
                else:
                    turnData = genTurnData_nbest(turn, labelJson)
                fileData["turns"].append(turnData)
                fileDatas.append(copy.deepcopy(fileData))

            tracker_outputs = mod_tracker.get_batch_new_state(fileDatas)
            for i in xrange(len(tracker_outputs)):
                del_none_val(tracker_outputs[i])
                if tagged:
                    tag_to_val(tracker_outputs[i], tag_dicts[i])
                res_dialogue["turns"].append(tracker_outputs[i])
            res["sessions"].append(res_dialogue)
            print "processed dialogue no.:", len(res["sessions"])

    # process by turn-level dialogue
    elif level == 'turn':

        batch_size = mod_tracker.batch_size

        fileDatas_all=gen_resdata(dataset,'nbest_tagged')
        # fileDatas_all = []
        # for call in dataset:
        #     fileData = {}
        #     fileData["turns"] = []
        #     fileData["session-id"] = call.log["session-id"]
        #     for turn, labelJson in call:
        #         turnData = genTurnData_nbest(turn, labelJson)
        #         fileData["turns"].append(turnData)
        #     fileDatas_all.append(fileData)
        
        batch_num = int(math.ceil(len(fileDatas_all[0]) / float(batch_size)))
        for j in xrange(batch_num):
            fileDatas0 = fileDatas_all[0][batch_size*j: batch_size*(j+1)]
            fileDatas1 = fileDatas_all[1][batch_size*j: batch_size*(j+1)]
            fileDatas=[]
            fileDatas.append(fileDatas0)
            fileDatas.append(fileDatas1)
            tracker_outputs = mod_tracker.get_turn_batch_state(fileDatas, feature_type)

            for i in xrange(len(fileDatas[0])):
                res_dialogue = dict()
                res_dialogue["session-id"] = fileDatas[0][i]["session-id"]
                res_dialogue["turns"] = tracker_outputs[i]
                for turn_output in res_dialogue["turns"]:
                    del_none_val(turn_output)
                res["sessions"].append(res_dialogue)
            print "processed dialogue no.:", len(res["sessions"])

    end_time = time.time()
    res['wall-time'] = end_time - start_time
    if tagged:
        baseline_json_file = 'baseline_%s_tagged.json'%dataset_name
    else:
        baseline_json_file = 'baseline_%s_dlstm.json'%dataset_name
    json.dump(res, open(baseline_json_file, 'wb'), indent=4)


if __name__ == '__main__':
    # #########################
    # Decide data set
    # #########################
    #dataset_name = 'dstc2_train'
    #dataroot = 'dstc2_traindev/data'

    #dataset_name = 'dstc2_test'
    #dataroot = 'dstc2_traindev/data'


    # #########################
    # Choose operation
    # #########################
    train_dstc2(0)
    ##gen_baseline_ground(dataset_name, dataroot)
    ##gen_baseline(dataset_name, dataroot, tagged=True)
    #gen_baseline(dataset_name, dataroot)
