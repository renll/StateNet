# -*- coding: utf-8 -*-

import time
import os
import sys
import pickle
import numpy as np

import mxnet as mx

from lectrack import LecTrack
from bucket_io import DSTSentenceIter
from turnbow_io import DSTTurnIter
from mat_io import MATTurnSentIter
from turnsent_io import DSTTurnSentIter

from mat_data import vocab,vocab1, ontologyDict

# 脚本所在位置
cur_dir = os.path.dirname(os.path.abspath(__file__))


class OfflineModel(object):
    def __init__(self, labelIdx,ini, context_type='cpu', config_dict=None):
        self.labelIdx = labelIdx
        self.ontology = ontologyDict
        self.vocab = vocab
        self.ini =ini
        self.vocab1 = vocab1
        # configuration
        config_dict = config_dict or {}
        self.nn_type            = config_dict.get('nn_type',    "lstm")
        self.model_dir          = config_dict.get('model_dir',  os.path.join(cur_dir, 'models'))
        self.train_json_file    = config_dict.get('train_json', os.path.join(cur_dir, 'train_custom.json'))
        self.dev_json_file      = config_dict.get('dev_json',   os.path.join(cur_dir, 'dev_custom.json'))
        self.test_json_file     = config_dict.get('test_json',  os.path.join(cur_dir, 'test_custom.json'))

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        print(mx.__version__)
        # ##################################
        self.label_index_list = ['goal_food', 'goal_pricerange', 'goal_name', 'goal_area', 'method', 'requested']
        self.num_label_dict = {
            'goal_food':        len(self.ontology["informable"]["food"]),
            'goal_pricerange':  len(self.ontology["informable"]["pricerange"]),
            'goal_name':        len(self.ontology["informable"]["name"]),
            'goal_area':        len(self.ontology["informable"]["area"]),
            'method':           len(self.ontology["method"]),
            'requested':        len(self.ontology["requestable"])
        }
        print self.num_label_dict
        # ##################################
        self.label_name=[]
        for i in labelIdx:
            self.label_name.append(self.label_index_list[i])
        self.num_label=[]
        for k in self.label_name:
            self.num_label.append(self.num_label_dict[k])
        
        self.batch_size = 32
        if self.nn_type in ['lstmn']:
            self.buckets = [3, 5, 8, 10, 13, 15, 18, 22, 26, 30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 90, 110, 150, 350]
        elif self.nn_type in ['cnn']:
            self.buckets = [10, 13, 15, 18, 22, 26, 30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 90, 110, 150, 350, 500]
        elif self.nn_type in ['reslstm','matlstm','bowlstm', 'cnnlstm', 'cnncnnlstm','doublelstm']:
            self.buckets = range(1, 31)
            # FIXME TODO
            self.batch_size = 32
        else:
            self.buckets = []
 
        if self.nn_type in ['reslstm','matlstm']:
            self.batch_size=32
            self.opt='rmsprop'
            self.drop=0.
            self.lr=0.0005
        elif self.nn_type in ['lstmcnn']:
            self.opt='sgd'
            self.drop=0.#0.2
            self.lr=0.001           
        else:
            if self.ini==1:
                self.opt='rmsprop'#rmsprop
                self.drop=0.#0.
                self.lr=0.0005#0.0005      
            else:
                self.opt='adam'#rmsprop
                self.drop=0.#0.
                self.lr=0.001#0.0005  
        # TODO be careful about "pretrain_embed"
        lectrack_config_dict = {
            "batch_size": self.batch_size,
            "input_size": len(self.vocab1),
            "num_label": self.num_label,
            "output_type": 'sigmoid' if 'requested' in self.label_name else 'softmax',
            "nn_type": self.nn_type,
            "context_type": context_type,
            "dropout":self.drop,#0.2
            "optimizer": self.opt,#adam
            "buckets": self.buckets,
            "pretrain_embed": False,
            "fix_embed": True,
            "num_lstm_o": 128,#256
            "num_hidden": 128,
            "learning_rate": self.lr,#0.01mat,0.001dlstm
            "enable_mask": False,
            "N": 1,
            "num_embed": 300
        }
        self.lectrack = LecTrack(lectrack_config_dict)
        self.init_states = self.lectrack.init_states
        self.data_components = self.lectrack.data_components
        if 'requested' in self.label_name:
            self.label_out = self.num_label_dict['requested']
        else:
            self.label_out = len(self.labelIdx)

        # ##################################
        if self.nn_type in ['bowlstm']:
            self.data_train = DSTTurnIter(self.train_json_file, self.labelIdx, self.vocab, self.buckets, self.batch_size,
                                            self.init_states, self.data_components, label_out=self.label_out)
            self.data_val = DSTTurnIter(self.dev_json_file, self.labelIdx, self.vocab, self.buckets, self.batch_size,
                                            self.init_states, self.data_components, label_out=self.label_out)
            self.data_test = DSTTurnIter(self.test_json_file, self.labelIdx, self.vocab, self.buckets, self.batch_size,
                                            self.init_states, self.data_components, label_out=self.label_out)
        elif self.nn_type in ['reslstm','matlstm','doublelstm']:
            self.max_nbest = self.lectrack.max_nbest
            self.max_sentlen = self.lectrack.max_sentlen
            if self.nn_type in ['doublelstm']:
                feature_type = 'sentbow'
            elif self.nn_type in ['reslstm','matlstm']:
                feature_type= 'bowbow'
            else:
                feature_type = 'sentsent'
            self.data_train = MATTurnSentIter(self.train_json_file, self.labelIdx, self.vocab, self.vocab1,self.buckets, self.batch_size, self.max_nbest, self.max_sentlen,
                                            self.init_states, self.data_components, label_out=self.label_out, feature_type=feature_type)
            self.data_val = MATTurnSentIter(self.dev_json_file, self.labelIdx, self.vocab, self.vocab1,self.buckets, self.batch_size, self.max_nbest, self.max_sentlen,
                                            self.init_states, self.data_components, label_out=self.label_out, feature_type=feature_type)
            self.data_test = MATTurnSentIter(self.test_json_file, self.labelIdx, self.vocab, self.vocab1,self.buckets, self.batch_size, self.max_nbest, self.max_sentlen,
                                            self.init_states, self.data_components, label_out=self.label_out, feature_type=feature_type)
        elif self.nn_type in ['cnncnnlstm','cnnlstm']:
            self.max_nbest = self.lectrack.max_nbest
            self.max_sentlen = self.lectrack.max_sentlen
            feature_type = 'sentsent'
            self.data_train = DSTTurnSentIter(self.train_json_file, self.labelIdx, self.vocab, self.buckets, self.batch_size, self.max_nbest, self.max_sentlen,
                                            self.init_states, self.data_components, label_out=self.label_out, feature_type=feature_type)
            self.data_val = DSTTurnSentIter(self.dev_json_file, self.labelIdx, self.vocab, self.buckets, self.batch_size, self.max_nbest, self.max_sentlen,
                                            self.init_states, self.data_components, label_out=self.label_out, feature_type=feature_type)
            self.data_test = DSTTurnSentIter(self.test_json_file, self.labelIdx, self.vocab, self.buckets, self.batch_size, self.max_nbest, self.max_sentlen,
                                            self.init_states, self.data_components, label_out=self.label_out, feature_type=feature_type)
        else:
            self.data_train = DSTSentenceIter(self.train_json_file, self.labelIdx, self.vocab, self.buckets, self.batch_size,
                                            self.init_states, self.data_components, label_out=self.label_out)
            self.data_val = DSTSentenceIter(self.dev_json_file, self.labelIdx, self.vocab, self.buckets, self.batch_size,
                                            self.init_states, self.data_components, label_out=self.label_out)
            self.data_test = DSTSentenceIter(self.test_json_file, self.labelIdx, self.vocab, self.buckets, self.batch_size,
                                            self.init_states, self.data_components, label_out=self.label_out)
# ##################################
        if self.lectrack.optimizer == 'adam':
            lrFactor=0.5
            epochSize=self.data_train.size
            step_epochs=[150]
            steps=[epochSize*x for x in step_epochs]
            scheduler= mx.lr_scheduler.MultiFactorScheduler(step=steps,factor=lrFactor)
            self.lectrack.model.init_optimizer(optimizer='adam',optimizer_params=(('learning_rate',self.lectrack.learning_rate),('clip_gradient',2.0),))#, ('lr_scheduler',scheduler),))
            #self.model.init_optimizer(optimizer='adam',optimizer_params=(('learning_rate',self.learning_rate), ('clip_gradient',5.0)))
        elif self.lectrack.optimizer == 'adagrad':
            self.lectrack.model.init_optimizer(optimizer='adagrad')
        elif self.lectrack.optimizer == 'rmsprop':
            self.lectrack.model.init_optimizer(optimizer='rmsprop',optimizer_params=(('learning_rate',self.lectrack.learning_rate),('centered',False), ))#('gamma1',0.95),('clip_gradient',10), ))
        elif self.lectrack.optimizer == 'sgd':
            lrFactor=0.5
            epochSize=self.data_train.size
            step_epochs=[30,80,130]
            steps=[epochSize*x for x in step_epochs]
            scheduler= mx.lr_scheduler.MultiFactorScheduler(step=steps,factor=lrFactor)
            self.lectrack.model.init_optimizer(optimizer='sgd',optimizer_params=(('learning_rate',self.lectrack.learning_rate),('lr_scheduler',scheduler),('momentum',0.9), ('clip_gradient',5.0), ('wd',0.0001), ))
        print '[INFO]: Using optimizer - %s' % self.lectrack.optimizer
       
        if self.ini==0: 
            params_path = os.path.join(self.model_dir, 'labelIdx-99-devbest.params' )
            if os.path.exists(params_path):
                self.lectrack.load_params(params_path)



        # ##################################
        if 'requested' in self.label_name:
            def customAcc(labels, preds):
                ret = 0
                if len(labels.shape) > len(preds.shape):
                    preds_true_shape = (self.batch_size, preds.shape[0]/self.batch_size) + preds.shape[1:]
                    preds = preds.reshape(preds_true_shape)
                for label, pred_label in zip(labels, preds):
                    pred_label = (pred_label + 0.5).astype('int32')
                    label = label.astype('int32')
                    ret += np.sum(np.all(np.equal(pred_label, label), axis=-1))
                return (ret, len(labels.reshape((-1, labels.shape[-1]))))
            self.metric = mx.metric.CustomMetric(customAcc)
        else:
            def customAcc(labels, preds):
                ret = 0
                l=0
                for label, pred_label in zip(labels, preds):
                   # if pred_label.shape != label.shape:
                    #    pred_label = np.argmax(pred_label, axis=-1)
                    pred_label = pred_label.astype('int32')
                    label = label.astype('int32')
                    ret += (pred_label == label).all(-1).sum()
                    l+= len((pred_label == label).all(-1)) 
                return (ret, l)
            self.metric = mx.metric.CustomMetric(customAcc,output_names=['softmax_output'],label_names=['softmax_label'])

    def custom_score(self, eval_data, eval_metric):
        eval_data.reset()
        eval_metric.reset()
        pad_count = 0
        
        for nbatch, eval_batch in enumerate(eval_data):
            label_shape = eval_batch.provide_label[0][1]
            pad_count += eval_batch.pad * label_shape[1]
            self.lectrack.model.forward(eval_batch, is_train=False)
            self.lectrack.model.update_metric(eval_metric, eval_batch.label)
        eval_metric.sum_metric -= pad_count
        eval_metric.num_inst -= pad_count
        return eval_metric.get_name_value()


    def offline_train(self, num_epoch=20):
        print '====== Train labelIdx-%d:' % 99# self.labelIdx
        print '[Start] ', time.strftime('%Y-%m-%d %H-%M', time.localtime(time.time()))
        
        if self.ini==1:
            model_name='labelIdx-%d-devbest.params' % 99
        else:
            model_name='labelIdx-%d-devbest.params' % 999

        best_dev_info = {
            "params_path": os.path.join(self.model_dir, model_name),#self.labelIdx),
            "epoch_num": -1,
            "acc": 0.0
        }
        best_test_info = {
            "params_path": os.path.join(self.model_dir, 'labelIdx-joint-testbest.params' ),
            "epoch_num": -1,
            "acc": 0.0
        }



        for i in range(num_epoch):
            print("=============BEGIN EPOCH %d==================="%i)
            for batch in self.data_train:
                self.lectrack.model.forward(batch, is_train=True)
                self.lectrack.model.update_metric(self.metric, batch.label)
                self.lectrack.model.backward()
                self.lectrack.model.update()
            print(self.metric.get())
            print '[Training over] ', time.strftime('%Y-%m-%d %H-%M', time.localtime(time.time()))

            #params_path = os.path.join(self.model_dir, 'labelIdx-%d-epoch-%d.params' % (self.labelIdx, i))
            #self.lectrack.model.save_params(params_path)
            self.data_train.reset()

            # evaluate on different data set and save best model automatically
            # FIXME using custom score function
            print 'train:', self.custom_score(self.data_train, self.metric)
            dev_score = self.custom_score(self.data_val, self.metric)
            print 'dev  :', dev_score
            test_score = self.custom_score(self.data_test, self.metric)
            print 'test :', test_score
            print '[Testing over] ', time.strftime('%Y-%m-%d %H-%M', time.localtime(time.time()))
            sys.stdout.flush()
            if dev_score[0][1] > best_dev_info["acc"]:
                best_dev_info["acc"] = dev_score[0][1]
                best_dev_info["epoch_num"] = i
                self.lectrack.model.save_params(best_dev_info["params_path"])
            if test_score[0][1] > best_test_info["acc"]:
                best_test_info["acc"] = test_score[0][1]
                best_test_info["epoch_num"] = i
                self.lectrack.model.save_params(best_test_info["params_path"])

            print 'devbest epoch: %s, acc: %s' % (best_dev_info['epoch_num'], best_dev_info['acc'])
            print 'testbest epoch: %s, acc: %s' % (best_test_info['epoch_num'], best_test_info['acc'])

        print '[End] ', time.strftime('%Y-%m-%d %H-%M',time.localtime(time.time()))
        print "="*50
        sys.stdout.flush()


    def offline_eval(self):
        print '====== Eval labelIdx-%d:' % 99
        print '[Start] ', time.strftime('%Y-%m-%d %H-%M', time.localtime(time.time()))

        # load pre-trained params
        params_path = os.path.join(self.model_dir, 'labelIdx-%d-devbest.params' % 999)
        self.lectrack.load_params(params_path)
        
        print 'train:', self.custom_score(self.data_train, self.metric)
        dev_score = self.custom_score(self.data_val, self.metric)
        print 'dev  :', dev_score
        test_score = self.custom_score(self.data_test, self.metric)
        print 'test :', test_score
        print '[Testing over] ', time.strftime('%Y-%m-%d %H-%M', time.localtime(time.time()))
        sys.stdout.flush()
           
        #print 'train:', self.lectrack.model.score(self.data_train, self.metric)
        #print 'dev  :', self.lectrack.model.score(self.data_val, self.metric)
        #print 'test :', self.lectrack.model.score(self.data_test, self.metric)

        #print '[End] ', time.strftime('%Y-%m-%d %H-%M',time.localtime(time.time()))


def evalAll():
    ctx = 'cpu'

    #tmp_model = OfflineModel(5, ctx)
    #tmp_model.offline_eval()
    for i in xrange(6):
        tmp_model = OfflineModel(i, ctx)
        tmp_model.offline_eval()

def trainAll():
    ctx = 'cpu'

    #tmp_model = OfflineModel(5, ctx)
    #tmp_model.offline_train()
    for i in xrange(6):
        tmp_model = OfflineModel(i, ctx)
        tmp_model.offline_train()


if __name__ == '__main__':
    pass
    #evalAll()
    #trainAll()
