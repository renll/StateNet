# -*- coding: utf-8 -*-

import sys
import copy
import math
import os
import json

from lectrack import LecTrack
from bucket_io import read_1best_dialog_content, default_text2id
from mat_io import read_nbest_dialog_content, text2bow
from turnbow_io import nbest_text2bow
from mat_data import genTurnData_nbest

from mat_data import vocab, vocab1, ontologyDict

# 脚本所在位置
cur_dir = os.path.dirname(os.path.abspath(__file__))


class ModTracker(object):
    """ModTracker implementation with LecTrack
    ------
    config_dict {
        nn_type: string, "lstm" or "blstm"
        for_train: boolean, whether to train the models during running
        pre_train: boolean, whether to load pre-trained models
        model_dir: string, dir of pre_train models
    }
    """
    def __init__(self, config_dict=None):
        self.ontology = ontologyDict
        

        # configuration
        config_dict = config_dict or {}
        self.context_type   = config_dict.get('context_type',   'cpu')
        self.nn_type        = config_dict.get('nn_type',        "lstm")
        self.batch_size     = config_dict.get('batch_size',     32)
        self.for_train      = config_dict.get('for_train',      False)
        self.pre_train      = config_dict.get('pre_train',      True)
        self.model_dir      = config_dict.get('model_dir',      os.path.join(cur_dir, 'models'))

        # ##################################
        self.label_index_list = ['goal_food', 'goal_pricerange', 'goal_name', 'goal_area']
        self.num_label_dict = {
            'goal_food':        len(self.ontology["informable"]["food"]),
            'goal_pricerange':  len(self.ontology["informable"]["pricerange"]),
            'goal_name':        len(self.ontology["informable"]["name"]),
            'goal_area':        len(self.ontology["informable"]["area"]),
            #'method':           len(self.ontology["method"]),
            #'requested':        len(self.ontology["requestable"])
        }

        # ##################################
        self.track_dict = {}
        self.vocab = {}
        for label_name, num_label in self.num_label_dict.items():
            #if label_name=='goal_food' or label_name=='goal_name':
            self.vocab[label_name]= vocab1
            #else:
            #    self.vocab[label_name]= vocab2
            lectrack_config_dict = {
                "input_size": len(self.vocab[label_name]),
                "num_label": num_label,
                "output_type": 'sigmoid' if label_name == 'requested' else 'softmax',
                "nn_type": self.nn_type,
                "batch_size": self.batch_size,
                "context_type": self.context_type,
            }
            print label_name
            print lectrack_config_dict["input_size"]
            self.track_dict[label_name] = LecTrack(lectrack_config_dict)

        # load pre-trained params if needed
        if self.pre_train == True:
            for label_index, label_name in enumerate(self.label_index_list):
                # TODO

                #params_path = os.path.join(self.model_dir, 'labelIdx-%d-final.params' % label_index)
                params_path = os.path.join(self.model_dir, 'labelIdx-%d-testbest.params' % label_index)

                print label_index,label_name
                self.track_dict[label_name].load_params(params_path)

        # about dialogue history
        self.history_index = 0
        self.history_label = {
            "turns": []
        }
        self.history_log = {
            "turns": []
        }
        # reset
        self.reset()

    def reset(self):
        """重置/初始化 各个成员变量"""
        self.turn = 0

        # TODO CALL ONLY WHEN NEEDED!!! save current dialogue history to file
        #self._saveHistory()

        if len(self.history_log["turns"]) > 0:
            self.history_index += 1
        self.history_label = {"turns": []}
        self.history_log = {"turns": []}

    def _saveHistory(self):
        """（需要的话再调用该函数）保存当前对话的历史到文件中"""
        save_dialog_dir = os.path.join(cur_dir, 'dst_history')
        if not os.path.exists(save_dialog_dir):
            os.makedirs(save_dialog_dir)
        if len(self.history_log["turns"]) > 0:
            label_path = os.path.join(save_dialog_dir, 'label-%d.json' % self.history_index)
            json.dump(self.history_label, open(label_path, 'w'), indent=4)
            log_path = os.path.join(save_dialog_dir, 'log-%d.json' % self.history_index)
            json.dump(self.history_log, open(log_path, 'w'), indent=4)

    def _updateHistory(self, dm_output, asr_output, us_live_goal):
        """更新当前对话（同一个session）的历史信息"""
        # transfer us_live_goal to specific format if needed
        tmp_us_live_goal = copy.deepcopy(us_live_goal)
        if isinstance(tmp_us_live_goal["requested-slots"], dict):
            tmp_us_live_goal["requested-slots"] = []
        # update history_label
        self.history_label["turns"].append(tmp_us_live_goal)

        # transfer dm_output format to specific format if needed
        log_output = {}
        if "dialog-acts" in dm_output:
            log_output = dm_output
        else:
            tmp_dm_output = []
            for act in dm_output:
                new_act = {
                    "act": act["act_type"],
                    "slots": []
                }
                if "slot_name" in act and "slot_val" in act:
                    new_act["slots"].append([act["slot_name"], act["slot_val"]])
                elif "slot_name" in act:
                    new_act["slots"].append(["slot", act["slot_name"]])
                tmp_dm_output.append(new_act)
            log_output["dialog-acts"] = tmp_dm_output

        # transfer asr_output score to log-score if needed
        log_input = {}
        if "live" in asr_output:
            log_input = asr_output
        else:
            tmp_asr_output = []
            for hyp in asr_output["gen+asr"]:
                tmp_asr_output.append({
                    "asr-hyp": hyp["asr-hyp"],
                    "score": math.log(hyp['score'])
                })
            log_input = {
                "live": {
                    "asr-hyps": tmp_asr_output
                }
            }

        # update history_log
        self.history_log["turns"].append({
            "input": log_input,
            "output": log_output
        })

    def _updateState(self, cur_state, cur_outputs, label_name, top_n=sys.maxint):
        """根据lstm的输出和当前label_name更新cur_state(原地修改), 该函数的调用不能覆盖其他不相关label的值"""
        def float_floor(num):
            return float('%0.4f'%(num-0.00005 if num-0.00005>0 else 0.0))

        key_list = label_name.split('_')
        if key_list[0] == 'requested':
            cur_state["requested-slots"] = {}
            for i in xrange(len(cur_outputs)-1):
                slot_name = self.ontology["requestable"][i]
                cur_state["requested-slots"][slot_name] = float_floor(cur_outputs[i])
                #cur_state["requested-slots"][slot_name] = cur_outputs[i]
        elif key_list[0] == 'method':
            cur_state["method-label"] = {}
            for i in xrange(len(cur_outputs)):
                tmp_key = self.ontology["method"][i]
                cur_state["method-label"][tmp_key] = float_floor(cur_outputs[i])
        elif key_list[0] == 'goal':
            # 只取概率最高的若干个和 "none" 作为可能的取值
            cur_pairs = [(cur_outputs[i], i) for i in xrange(len(cur_outputs))]
            max_part = sorted(cur_pairs[:-1], key=lambda x:x[0], reverse=True)[:top_n]
            max_part += [cur_pairs[-1]]
            tmp_sum = sum([p[0] for p in max_part])
            max_part = [(float_floor(p[0]/tmp_sum), p[1]) for p in max_part]
            slot_name = key_list[1]
            cur_state["goal-labels"][slot_name] = {}
            for p in max_part:
                tmp_key = self.ontology["informable"][slot_name][p[1]] if p[1] != len(cur_outputs)-1 else 'none'
                if p[0] > 0.0:
                    cur_state["goal-labels"][slot_name][tmp_key] = p[0]

    def get_new_state(self, dm_output, asr_output, pre_state=None, us_live_goal=None):
        """[用于线上]生成新的state，调用该方法会影响self的状态，即当前输入的turn会被认为与之前的turn相关"""
        self.turn += 1
        cur_state = {
            "goal-labels": {},
            "method-label": {
                "none": 1.0
            },
            "requested-slots": {}
        }
        self._updateHistory(dm_output, asr_output, us_live_goal)

        # construct data format for generating DataBatch
        fileData = {}
        fileData["turns"] = []
        for i in xrange(len(self.history_label["turns"])):
            turnData = genTurnData_nbest(self.history_log["turns"][i], self.history_label["turns"][i])
            fileData["turns"].append(turnData)

        # update state
        for label_index, label_name in enumerate(self.label_index_list):
            dialog_sentences, dialog_scores, dialog_labels = read_1best_dialog_content(fileData, label_index)
            cur_sentence, cur_score, cur_label = dialog_sentences[-1], dialog_scores[-1], dialog_labels[-1]

            cur_sentence = default_text2id(cur_sentence, self.vocab[label_name])
            assert len(cur_sentence) > 0 and len(cur_sentence) == len(cur_score)

            tmp_label_out = len(cur_label) if self.track_dict[label_name].output_type == 'sigmoid' else 1
            data_batch = self.track_dict[label_name].oneSentenceBatch(cur_sentence, cur_score, cur_label, tmp_label_out)
            cur_outputs = self.track_dict[label_name].predict(data_batch)[0]
            cur_outputs = cur_outputs[0].asnumpy()
            self._updateState(cur_state, cur_outputs, label_name, top_n=5)

        # remove "signature" from requested_slots
        if "signature" in cur_state["requested-slots"]:
            del cur_state["requested-slots"]["signature"]

        return cur_state

    def get_batch_new_state(self, fileDatas):
        """[用于线下]同时生成多个新state，调用该方法不会影响self的状态(注意样本的个数不能超过LecTrack的batch_size)
        batch的每个example是：包含若干个turn的一个对话拼接成一个长的句子，输出是：对于每个example有一个预测值"""
        assert(len(fileDatas) <= self.batch_size)
        tracker_outputs = []
        for i in xrange(len(fileDatas)):
            tracker_outputs.append({
                "goal-labels": {},
                "method-label": {
                    "none": 1.0
                },
                "requested-slots": {}
            })

        for label_index, label_name in enumerate(self.label_index_list):
            sentences, scores, labels = [], [], []
            for fileData in fileDatas:
                dialog_sentences, dialog_scores, dialog_labels = read_1best_dialog_content(fileData, label_index)
                cur_sentence, cur_score, cur_label = dialog_sentences[-1], dialog_scores[-1], dialog_labels[-1]

                cur_sentence = default_text2id(cur_sentence, self.vocab[label_name])
                assert len(cur_sentence) > 0 and len(cur_sentence) == len(cur_score)

                sentences.append(cur_sentence)
                scores.append(cur_score)
                labels.append(cur_label)
                tmp_label_out = len(cur_label) if self.track_dict[label_name].output_type == 'sigmoid' else 1

            data_batch = self.track_dict[label_name].multiWordSentBatch(sentences, scores, labels, tmp_label_out)
            outputs = self.track_dict[label_name].predict(data_batch)[0]

            for i in xrange(len(tracker_outputs)):
                cur_outputs = outputs[i].asnumpy()
                self._updateState(tracker_outputs[i], cur_outputs, label_name, top_n=10)

        # remove "signature" from requested_slots
        for i in xrange(len(tracker_outputs)):
            if "signature" in tracker_outputs[i]["requested-slots"]:
                del tracker_outputs[i]["requested-slots"]["signature"]

        return tracker_outputs

    def get_turn_batch_state(self, fileDatas, feature_type='bow'):
        """[用于线下]同时生成多个新state，调用该方法不会影响self的状态
        batch的每个example是：包含若干个单独turn的一个对话每个turn是一个sentence，输出是：对于每个example的每个turn都有一个预测值
        Parameters:
            feature_type, 仅当level=turn时有效，取值['bow', 'sentsent', 'sentbow'], 表明turn级别的特征是如何生成的"""
        tracker_outputs = []
        for i in xrange(len(fileDatas[0])):
            tracker_outputs.append([])
            for j in xrange(len(fileDatas[0][i]["turns"])):
                tracker_outputs[i].append({
                    "goal-labels": {},
                    "method-label": {
                        "none": 1.0
                    },
                    "requested-slots": {}
                })


        for label_index, label_name in enumerate(self.label_index_list):
            # generate data for batch use accroding to current feature_type
            if feature_type == 'bow':
                sentences, acts, labels = [], [], []
                for fileData in fileDatas:
                    dialog_sentences, dialog_scores, machine_acts, dialog_labels = read_nbest_dialog_content(fileData, label_index)
                    sentence_turn, act_turn, label_turn = [], [], []
                    for turn_id in xrange(len(dialog_sentences)):
                        cur_sentence = nbest_text2bow(dialog_sentences[turn_id], dialog_scores[turn_id], self.vocab[label_name])
                        cur_act = text2bow(machine_acts[turn_id], self.vocab[label_name])
                        cur_label = dialog_labels[turn_id]
                        tmp_label_out = len(cur_label) if self.track_dict[label_name].output_type == 'sigmoid' else 1
                        sentence_turn.append(cur_sentence)
                        act_turn.append(cur_act)
                        label_turn.append(cur_label)
                    sentences.append(sentence_turn)
                    acts.append(act_turn)
                    labels.append(label_turn)  
                data_batch = self.track_dict[label_name].multiTurnBowBatch(sentences, acts, labels, tmp_label_out)

            elif feature_type in ['sentsent', 'sentbow', 'bowsent', 'bowbow']:
                def turn_read_content(fileDatas,dataIdx,feature_type):
                    sentences, acts, scores, labels = [], [], [], []

                    for fileData in fileDatas[dataIdx]:
                        dialog_sentences, dialog_scores, machine_acts, dialog_labels = read_nbest_dialog_content(fileData, label_index)
                        
                        sentence_turn, act_turn, score_turn, label_turn = [], [], [], []

                        for turn_id in xrange(len(dialog_sentences)):
                            cur_sentence = []
                            # user sentence feature
                            for nbest_id in range(len(dialog_sentences[turn_id])):
                                if feature_type in ['sentsent', 'sentbow']:
                                    cur_sentbest = default_text2id(dialog_sentences[turn_id][nbest_id], vocab)
                                elif feature_type in ['bowsent', 'bowbow']:
                                    cur_sentbest = text2bow(dialog_sentences[turn_id][nbest_id], self.vocab[label_name])
                                cur_sentence.append(cur_sentbest)
                            # sys act feature
                            if feature_type in ['sentbow', 'bowbow']:
                                cur_act = text2bow(machine_acts[turn_id], self.vocab[label_name])
                            elif feature_type in ['sentsent', 'bowsent']:
                                cur_act = default_text2id(machine_acts[turn_id], self.vocab[label_name])
                            cur_score = dialog_scores[turn_id]
                            cur_label = dialog_labels[turn_id]
                            tmp_label_out = len(cur_label) if self.track_dict[label_name].output_type == 'sigmoid' else 1
                            sentence_turn.append(cur_sentence)
                            act_turn.append(cur_act)
                            score_turn.append(cur_score)
                            label_turn.append(cur_label)

                        sentences.append(sentence_turn)
                        scores.append(score_turn)
                        acts.append(act_turn)
                        labels.append(label_turn)
                    return sentences, scores, acts, labels,tmp_label_out

                sentences, scores, acts, labels,tmp_label_out = turn_read_content(fileDatas,0,feature_type)
                sentences1, scores1, acts1, labels1,tmp_label_out= turn_read_content(fileDatas,1,feature_type)
                sentences0=[]
                for i in range(len(sentences)):
                    dialog0=[]
                    for j in range(len(sentences[i])):
                        sent=[]
                        sent.append(sentences[i][j])
                        sent.append(sentences1[i][j]) 
                        dialog0.append(sent)
                    sentences0.append(dialog0)
                sentences=sentences0
                act0=[]
                for i in range(len(acts)):
                    dialog0=[]
                    for j in range(len(acts[i])):
                        act=[]
                        act.append(acts[i][j])
                        act.append(acts1[i][j])
                        dialog0.append(act)
                    act0.append(dialog0)
                acts=act0

                data_batch = self.track_dict[label_name].multiTurnBatch(label_index,sentences, acts, scores, labels, tmp_label_out, vocab, self.vocab[label_name],feature_type)

            outputs = self.track_dict[label_name].predict(data_batch)[0]
            if outputs.shape[0] != self.batch_size:
                outputs = outputs.reshape((self.batch_size, -1,) + outputs.shape[1:])

            for i in xrange(len(tracker_outputs)):
                for j in xrange(len(tracker_outputs[i])):
                    cur_outputs = outputs[i][j].asnumpy()
                    self._updateState(tracker_outputs[i][j], cur_outputs, label_name, top_n=10)

        # remove "signature" from requested_slots
        for i in xrange(len(tracker_outputs)):
            for j in xrange(len(tracker_outputs[i])):
                if "signature" in tracker_outputs[i][j]["requested-slots"]:
                    del tracker_outputs[i][j]["requested-slots"]["signature"]

        return tracker_outputs


if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    mod_tracker = ModTracker()

    tmp_us_live_goal = {
        "goal-labels": {
            "food": "french",
            "name": "hotel du vin and bistro"
        },
        "method-label": "byname",
        "requested-slots": []
    }
    tmp_dm_output = [
        #{
        #    "slot_name": "chinese",
        #    "act_type": "inform",
        #    "slot_val": "food",
        #}
        #{
        #    "slot_name": "pricerange",
        #    "act_type": "request",
        #}
        {
            "act_type": "welcomemsg",
        }
    ]
    tmp_asr_output = {
        "gen+asr": [
            {
                #"asr-hyp": "i dont care",
                "asr-hyp": "expensive restaurant in west",
                #"asr-hyp": "what about the price",
                "score": 0.9
            },
            {
                #"asr-hyp": "any will do",
                "asr-hyp": "i am",
                "score": 0.1
            }
        ],
    }

    tmp_output = mod_tracker.get_new_state(tmp_dm_output, tmp_asr_output, pre_state=None, us_live_goal=tmp_us_live_goal)
    print '------mod_tracker--------'
    print tmp_output
    print json.dumps(tmp_output, indent=4)
    print '------------------------------------------------'
