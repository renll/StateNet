# -*- coding: utf-8 -*-

#import sys
import numpy as np
import mxnet as mx

from DSTmodels import lstm_unroll
from DSTmodels import lstmcnn_unroll
from DSTmodels import blstm_unroll
from DSTmodels import lstmn_unroll
from DSTmodels import text_cnn
from DSTmodels import bowlstm_unroll
from DSTmodels import cnnlstm_unroll
from DSTmodels import cnncnnlstm_unroll
from DSTmodels import doublelstm_unroll
from DSTmodels import reslstm_unroll
from bucket_io import SimpleBatch
from bucket_io import default_text2id
from mat_data import ontologyDict


class LecTrack(object):
    """LecTrack implementation, config_dict:
        output_type: can be one of ['softmax', 'sigmoid']
        N: number of cpu/gpu cores
        pretrain_embed: whether to use pre-trained word embedding
        embed_matrix: pre-trained embed_matrix file
        fix_embed: whether to keep embed_matrix unchanged while training
    """
    def __init__(self, config_dict):
        # Configuration
        self.input_size         = config_dict.get('input_size')
        self.num_label          = config_dict.get('num_label')
        self.nn_type            = config_dict.get('nn_type',        'lstm')
        self.output_type        = config_dict.get('output_type',    'softmax')
        self.context_type       = config_dict.get('context_type',   'cpu')
        self.dropout            = config_dict.get('dropout',        0.)
        self.batch_size         = config_dict.get('batch_size',     32)
        self.optimizer          = config_dict.get('optimizer',      'adam')
        self.initializer        = config_dict.get('initializer',    'xavier')

        # Configurations that usually does not need to tune
        self.N                  = config_dict.get('N',              1)
        self.enable_mask        = config_dict.get('enable_mask',    False)
        self.num_embed          = config_dict.get('num_embed',      300)
        self.num_lstm_layer     = config_dict.get('num_lstm_layer', 1)
        self.num_lstm_o         = config_dict.get('num_lstm_o',     128)
        self.num_hidden         = config_dict.get('num_hidden',     128)
        self.learning_rate      = config_dict.get('learning_rate',  0.01)
        self.pretrain_embed     = config_dict.get('pretrain_embed', False)
        self.embed_matrix       = config_dict.get('embed_matrix',   'embed_mat.npy')
        self.fix_embed          = config_dict.get('fix_embed',      True)
        self.buckets            = config_dict.get('buckets',        [])

        # ##################################
        if self.nn_type in ['lstmn']:
            self.buckets = [3, 5, 8, 10, 13, 15, 18, 22, 26, 30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 90, 110, 150, 350]
        elif self.nn_type in ['cnn']:
            self.buckets = [10, 13, 15, 18, 22, 26, 30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 90, 110, 150, 350, 500]
        elif self.nn_type in ['reslstm','matlstm','bowlstm', 'cnnlstm', 'cnncnnlstm','doublelstm']:
            self.buckets = range(1, 31)
        else:
            self.buckets = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 500]
        self.default_bucket_key = 500 if len(self.buckets) == 0 else max(self.buckets)
        print '[INFO]: nn_type is %s, buckets are %s' % (self.nn_type, str(self.buckets))

        self.batch_size *= self.N
        if self.context_type == 'gpu':
            self.contexts = [mx.context.gpu(i) for i in range(self.N)]
        else:
            self.contexts = [mx.context.cpu(i) for i in range(self.N)]

        # ##################################
        if self.nn_type == "lstm":
            self.nn_unroll = lstm_unroll
            self.data_components = [('data', (self.batch_size, self.default_bucket_key)), \
                                    ('score', (self.batch_size, self.default_bucket_key))]
            self.init_c = [('l%d_init_c'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            self.init_h = [('l%d_init_h'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            self.init_states = self.init_c + self.init_h
        elif self.nn_type == "lstmcnn":
            self.nn_unroll = lstmcnn_unroll
            self.data_components = [('data', (self.batch_size, self.default_bucket_key)),]# \
                                    #('score', (self.batch_size, self.default_bucket_key))]
            self.init_c = [('l%d_init_c'%l, (self.batch_size, self.num_embed)) for l in range(self.num_lstm_layer)]
            self.init_h = [('l%d_init_h'%l, (self.batch_size, self.num_embed)) for l in range(self.num_lstm_layer)]
            self.init_states = self.init_c + self.init_h
        elif self.nn_type == "blstm":
            self.nn_unroll = blstm_unroll
            self.data_components = [('data', (self.batch_size, self.default_bucket_key)), \
                                    ('score', (self.batch_size, self.default_bucket_key))]
            self.forward_init_c = [('forward_l%d_init_c'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            self.forward_init_h = [('forward_l%d_init_h'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            self.backward_init_c = [('backward_l%d_init_c'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            self.backward_init_h = [('backward_l%d_init_h'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            self.init_states = self.forward_init_c + self.forward_init_h + self.backward_init_c + self.backward_init_h
        elif self.nn_type == "lstmn":
            self.nn_unroll = lstmn_unroll
            self.data_components = [('data', (self.batch_size, self.default_bucket_key)), \
                                    ('score', (self.batch_size, self.default_bucket_key))]
            self.init_c = [('l%d_init_c'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            self.init_h = [('l%d_init_h'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            self.init_states = self.init_c + self.init_h
        elif self.nn_type == "cnn":
            self.nn_unroll = text_cnn
            self.data_components = [('data', (self.batch_size, self.default_bucket_key))]
            self.init_states = []
        elif self.nn_type == "bowlstm":
            self.nn_unroll = bowlstm_unroll
            self.data_components = [('data', (self.batch_size, self.default_bucket_key, self.input_size)), \
                                    ('data_act', (self.batch_size, self.default_bucket_key, self.input_size))]
            self.init_c = [('l%d_init_c'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            self.init_h = [('l%d_init_h'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            self.init_states = self.init_c + self.init_h
        elif self.nn_type == "cnnlstm":
            self.nn_unroll = cnnlstm_unroll
            self.max_nbest = 10 # the maximum number of nbest asr in train/dev/test is 10
            self.max_sentlen = 30 # the maximum length of user utterance in train/dev/test is 23
            self.data_components = [('data', (self.batch_size, self.default_bucket_key,self.max_nbest,self.max_sentlen)), \
                                    ('data_act', (self.batch_size, self.default_bucket_key, self.max_sentlen)), \
                                    ('score', (self.batch_size, self.default_bucket_key, self.max_nbest))]
            self.init_c = [('l%d_init_c'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            self.init_h = [('l%d_init_h'%l, (self.batch_size, self.num_label)) for l in range(self.num_lstm_layer)]
            self.init_states = self.init_c + self.init_h
        elif self.nn_type == "reslstm":
            self.numM=2
            self.nn_unroll = reslstm_unroll
            self.max_nbest = 10 # the maximum number of nbest asr in train/dev/test is 10
            self.max_sentlen = 30 # the maximum length of user utterance in train/dev/test is 23
            self.data_components = [('data', (self.batch_size, self.default_bucket_key, 2,self.input_size)), \
                                    ('data_act', (self.batch_size, self.default_bucket_key,2, self.input_size))]
                                   # ('score', (self.batch_size, self.default_bucket_key, self.max_nbest))]
            self.init_c = [('l%d_init_c'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            self.init_h = [('l%d_init_h'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            self.init_m=[('m_init_m%d'%i,(self.batch_size,2,self.input_size))for i in range(self.numM)]
            #self.init_con=[('one',(self.batch_size,1,self.num_label))]
            self.init_states = self.init_m +self.init_c + self.init_h#+self.init_con

        elif self.nn_type == "doublelstm":
            self.nn_unroll = doublelstm_unroll
            self.max_nbest = 10 # the maximum number of nbest asr in train/dev/test is 10
            self.max_sentlen = 30 # the maximum length of user utterance in train/dev/test is 23
            
            self.numM=2

            val_comp=[]
            for i,nv in enumerate(self.num_label): 
                val_comp.append(('value_%d'%i,(self.batch_size,nv,300)))
            self.data_components = [('data', (self.batch_size, self.default_bucket_key, self.max_sentlen,300)), \
                                    ('data_act', (self.batch_size, self.default_bucket_key, self.input_size)),\
                                    ('slot',(self.batch_size,len(self.num_label),300)),\
                                    ]+val_comp
                                    
                                    #('value',(self.batch_size,self.num_label,300)) ] 
                                   # ('score', (self.batch_size, self.default_bucket_key, self.max_nbest))]
            #self.forward_init_c = [('forward_l%d_init_c'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            #self.forward_init_h = [('forward_l%d_init_h'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            self.backward_init_c = [('backward_l%d_init_c'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            self.backward_init_h = [('backward_l%d_init_h'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            #self.init_m=[('m_init_m%d'%i,(self.batch_size,2,128))for i in range(self.numM)]
            self.init_states = self.backward_init_c + self.backward_init_h#+self.init_m
        elif self.nn_type == "cnncnnlstm":
            self.nn_unroll = cnncnnlstm_unroll
            self.max_nbest = 10 # the maximum number of nbest asr in train/dev/test is 10
            self.max_sentlen = 35 # the maximum length of user utterance in train/dev/test is 23
            self.data_components = [('data', (self.batch_size, self.default_bucket_key, self.max_nbest, self.max_sentlen)), \
                                    ('data_act', (self.batch_size, self.default_bucket_key, self.max_sentlen)), \
                                    ('score', (self.batch_size, self.default_bucket_key, self.max_nbest))]
            self.init_c = [('l%d_init_c'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            self.init_h = [('l%d_init_h'%l, (self.batch_size, self.num_lstm_o)) for l in range(self.num_lstm_layer)]
            self.init_states = self.init_c + self.init_h

        if self.enable_mask and self.nn_type in ['reslstm','matlstm','lstm', 'blstm', 'lstmn']:
            self.data_components += [('data_mask_len', (self.batch_size,))]
            print '[INFO]: Enable data mask'
        self.default_provide_data = self.data_components + self.init_states

        tmp_label_out = self.num_label if self.output_type == 'sigmoid' else len(self.num_label)
        if self.nn_type in ['bowlstm','reslstm','matlstm', 'cnnlstm', 'cnncnnlstm','doublelstm']:
            self.default_provide_label = [('softmax_label', (self.batch_size, self.default_bucket_key, tmp_label_out))]
        else:
            self.default_provide_label = [('softmax_label', (self.batch_size, tmp_label_out))]

        # ##################################
        self.model = mx.mod.BucketingModule(
                sym_gen = self.sym_gen,
                default_bucket_key = self.default_bucket_key,
                context = self.contexts)
        self.model.bind(data_shapes = self.default_provide_data, label_shapes = self.default_provide_label)

        # ##################################
        if self.initializer == 'xavier':
            default_init = mx.init.Xavier(magnitude=2.)
        elif self.initializer == 'uniform':
            default_init = mx.init.Uniform(0.1)
        print '[INFO]: Using initializer - %s' % self.initializer

        if self.pretrain_embed and self.embed_matrix:
            print '[INFO]: Using pre-trained word embedding.'
            embed_weight = np.load(self.embed_matrix)
            #print len(embed_weight) 
            init = mx.initializer.Load(param={"embed_weight": embed_weight}, default_init=default_init, verbose=True)

            self.model.init_params(initializer=init) #aux_params=auxDic)
        else:
            if self.pretrain_embed and not self.embed_matrix:
                print '[WARNNING]: Pre-trained word embedding is not used.'
                print '[WARNNING]: Because: pretrain_embed is True while embed_matrix is not given.'
            #auxDic={}
            #auxDic['multn']=2.0
            #auxDic['one']=mx.nd.full((32,1,self.num_label),0.25)
            #cons=mx.init.Load(auxDic)
            #init=mx.init.Mixed(['one','.*'],[cons,default_init])
            gamma=mx.nd.full((512,),0.1)
            gamma1=mx.nd.full((128,),0.1)
            init = mx.initializer.Load(param={"g50_gamma": gamma,"g60_gamma":gamma,"g70_gamma":gamma1}, default_init=default_init, verbose=True)
            self.model.init_params(initializer=default_init)#mx.init.MSRAPrelu())


    def sym_gen(self, seq_len):
        if self.nn_type in ['lstm','lstmcnn', 'blstm', 'lstmn']:
            return self.nn_unroll(
                num_lstm_layer  = self.num_lstm_layer,
                seq_len         = seq_len,
                input_size      = self.input_size,
                num_hidden      = self.num_hidden,
                num_embed       = self.num_embed,
                num_lstm_o      = self.num_lstm_o,
                num_label       = self.num_label,
                output_type     = self.output_type,
                dropout         = self.dropout,
                fix_embed       = self.fix_embed,
                enable_mask     = self.enable_mask
            )
        elif self.nn_type in ['cnn']:
            return self.nn_unroll(
                seq_len         = seq_len,
                num_embed       = self.num_embed,
                input_size      = self.input_size,
                num_label       = self.num_label,
                filter_list     = [3, 4, 5],
                num_filter      = 200,
                output_type     = self.output_type,
                dropout         = self.dropout,
                fix_embed       = self.fix_embed
            )
        elif self.nn_type in ['bowlstm']:
            return self.nn_unroll(
                num_lstm_layer  = self.num_lstm_layer,
                seq_len         = seq_len,
                input_size      = self.input_size,
                num_hidden      = self.num_hidden,
                num_embed       = self.num_embed,
                num_lstm_o      = self.num_lstm_o,
                num_label       = self.num_label,
                output_type     = self.output_type,
                dropout         = self.dropout
            )
        elif self.nn_type in ['reslstm','matlstm','cnnlstm', 'cnncnnlstm','doublelstm']:
            return self.nn_unroll(
                num_lstm_layer  = self.num_lstm_layer,
                seq_len         = seq_len,
                input_size      = self.input_size,
                num_hidden      = self.num_hidden,
                num_embed       = self.num_embed,
                num_lstm_o      = self.num_lstm_o,
                num_label       = self.num_label,
                filter_list     = [3, 4, 5],
                num_filter      = 100,
                max_nbest       = self.max_nbest,
                max_sentlen     = self.max_sentlen,
                output_type     = self.output_type,
                dropout         = self.dropout
            )

    def load_params(self, params_path):
        self.model.load_params(params_path)

    def save_params(self, params_path):
        self.model.save_params(params_path)

    def train(self, data_batch):
        self.model.forward(data_batch)
        self.model.backward()
        self.model.update()

    def predict(self, data_batch):
        self.model.forward(data_batch, is_train=False)
        return self.model.get_outputs()

    def getMatchKey(self, sentence_len):
        if len(self.buckets) > 0:
            for key in self.buckets:
                if key >= sentence_len:
                    return key
        return sentence_len

    def oneSentenceBatch(self, cur_sentence, cur_score, cur_label, label_out):
        cur_bucket_key = self.getMatchKey(len(cur_sentence))

        data = np.zeros((self.batch_size, cur_bucket_key))
        data_mask_len = np.zeros((self.batch_size, ))
        data_score = np.zeros((self.batch_size, cur_bucket_key))
        label = np.zeros((self.batch_size, label_out))
        data[:, :len(cur_sentence)] = cur_sentence
        data_mask_len[:] = len(cur_sentence)
        data_score[:, :len(cur_score)] = cur_score
        label[:, :label_out] = cur_label

        data_names = [x[0] for x in self.default_provide_data]
        init_state_arrays = [mx.nd.zeros(x[1]) for x in self.init_states]
        data_all = [mx.nd.array(data)] + init_state_arrays
        if 'score' in data_names:
            data_all += [mx.nd.array(data_score)]
        if 'data_mask_len' in data_names:
            data_all += [mx.nd.array(data_mask_len)]

        label_names = ['softmax_label']
        label_all = [mx.nd.array(label)]

        data_batch = SimpleBatch(data_names, data_all, label_names, label_all, cur_bucket_key)
        return data_batch

    def multiWordSentBatch(self, sentences, scores, labels, label_out):
        assert(len(sentences) <= self.batch_size)
        cur_bucket_key = self.getMatchKey(max([len(s) for s in sentences]))

        data = np.zeros((self.batch_size, cur_bucket_key))
        data_mask_len = np.zeros((self.batch_size, ))
        data_score = np.zeros((self.batch_size, cur_bucket_key))
        label = np.zeros((self.batch_size, label_out))
        for i in xrange(len(sentences)):
            data[i, :len(sentences[i])] = sentences[i]
            data_mask_len[i] = len(sentences[i])
            data_score[i, :len(scores[i])] = scores[i]
            label[i, :label_out] = labels[i]

        data_names = [x[0] for x in self.default_provide_data]
        init_state_arrays = [mx.nd.zeros(x[1]) for x in self.init_states]
        data_all = [mx.nd.array(data)] + init_state_arrays
        if 'score' in data_names:
            data_all += [mx.nd.array(data_score)]
        if 'data_mask_len' in data_names:
            data_all += [mx.nd.array(data_mask_len)]

        label_names = ['softmax_label']
        label_all = [mx.nd.array(label)]

        data_batch = SimpleBatch(data_names, data_all, label_names, label_all, cur_bucket_key)
        return data_batch

    def multiTurnBowBatch(self, sentences, acts, labels, label_out):
        assert(len(sentences) <= self.batch_size)
        cur_bucket_key = self.getMatchKey(max([len(s) for s in sentences]))

        data = np.zeros((self.batch_size, cur_bucket_key, self.input_size))
        data_act = np.zeros((self.batch_size, cur_bucket_key, self.input_size))
        label = np.zeros((self.batch_size, cur_bucket_key, label_out))
        for i in xrange(len(sentences)):
            for j in xrange(len(sentences[i])):
                data[i, j, :len(sentences[i][j])] = sentences[i][j]
                data_act[i, j, :len(acts[i][j])] = acts[i][j]
                label[i, j, :label_out] = labels[i][j]

        data_names = [x[0] for x in self.default_provide_data]
        init_state_arrays = [mx.nd.zeros(x[1]) for x in self.init_states]
        data_all = [mx.nd.array(data), mx.nd.array(data_act)]
        data_all += init_state_arrays

        label_names = ['softmax_label']
        label_all = [mx.nd.array(label)]

        data_batch = SimpleBatch(data_names, data_all, label_names, label_all, cur_bucket_key)
        return data_batch


    def multiTurnBatch(self, labelIdx,sentences, acts, scores, labels, label_out, vocab, vocab1,feature_type='bowbow'):
        assert(len(sentences) <= self.batch_size)
        print len(vocab)
        
        print len(vocab1)
        cur_bucket_key = self.getMatchKey(max([len(s) for s in sentences]))

        padding_id = vocab['</s>']
        len_sent = self.max_sentlen if feature_type in ['sentsent', 'sentbow'] else len(vocab)
        len_act_sent = self.max_sentlen if feature_type in ['sentsent', 'bowsent'] else len(vocab1)
        
        embed_weight = mx.nd.array(np.load('embed_vN3.npy'))
        # convert data into ndarrays for better speed during training
       
        slotsent="food pricerange name area" 
        slota=default_text2id(slotsent, vocab)
        slotarr=slotsent.split()
        #print slota
        
        val_len=len(ontologyDict[u'informable'][slotarr[labelIdx]])

        vl=[]
        for key in ontologyDict[u'informable'][slotarr[labelIdx]]:
            #print key
            v=default_text2id(key,vocab)
            tmp=mx.nd.array(v)
            tmp= mx.nd.Embedding(data=tmp, input_dim=len(vocab), weight=embed_weight, output_dim=300, name='embed')
            tmp=mx.nd.sum(tmp,axis=0)
            v=tmp.asnumpy()
            vl.append(v)
        vl=np.asarray(vl)
        #print vl
        #print len(vl)
    
        tmp=mx.nd.array([slota[labelIdx]])
        tmp= mx.nd.Embedding(data=tmp, input_dim=len(vocab), weight=embed_weight, output_dim=300, name='embed')
        slota=tmp.asnumpy()

       

        value=np.zeros((self.batch_size,val_len,300)) 
        slot=np.zeros((self.batch_size,300)) 
        for i in range(self.batch_size):
            slot[i]=slota                  
            value[i]=vl
 




        datatmp = np.full((self.batch_size, cur_bucket_key,2, self.max_nbest, len_sent), padding_id,dtype=np.double)
        data_act = np.full((self.batch_size, cur_bucket_key,2, len_act_sent), padding_id,dtype=np.double)
        data_score = np.zeros((self.batch_size, cur_bucket_key, self.max_nbest))
        label = np.zeros((self.batch_size, cur_bucket_key, label_out))

        data = np.full((self.batch_size, cur_bucket_key,2, len_sent, 300), padding_id,dtype=np.double)
    
        for i_diag in range(len(sentences)):
            for i_turn in range(len(sentences[i_diag])):
                act = acts[i_diag][i_turn]
                for i in range(2):
                    data_act[i_diag, i_turn,i, :len(act[i])] = act[i]
                label[i_diag, i_turn, :] = labels[i_diag][i_turn]
                # be careful that, here, max_nbest can be smaller than current turn nbest number. extra-best will be truncated.
                for i_data in range(2):
                    tempsent=[]    
                    for i_nbest in range(min(len(sentences[i_diag][i_turn][i_data]), self.max_nbest)):
                        sentence = sentences[i_diag][i_turn][i_data][i_nbest]
                        datatmp[i_diag, i_turn, i_data,i_nbest, :len(sentence)] = sentence
                        tmp=mx.nd.array(datatmp[i_diag, i_turn, i_data,i_nbest])
                        tmp= mx.nd.Embedding(data=tmp, input_dim=len(vocab), weight=embed_weight, output_dim=300, name='embed')
                        sentence=tmp.asnumpy()
                        score = scores[i_diag][i_turn][i_nbest]
                        #preprocess
                        sent =sentence*score 
                        tempsent.append(sent)
                        data_score[i_diag, i_turn, i_nbest] = score
                    tempsent=np.asarray(tempsent)
                    scoredsent=np.sum(tempsent,axis=0)
                    #scoredsent=scoredsent*2-1 
                    data[i_diag, i_turn, i_data] = scoredsent




        data_names = [x[0] for x in self.default_provide_data]
        init_state_arrays = [mx.nd.zeros(x[1]) for x in self.init_states]
        data_all = [mx.nd.array(data), mx.nd.array(data_act)]
        if 'score' in data_names:
            data_all += [mx.nd.array(data_score)]
        if 'slot' in data_names:
            data_all += [mx.nd.array(slot)]
        if 'value' in data_names:
            data_all += [mx.nd.array(value)]

        
        data_all += init_state_arrays

        label_names = ['softmax_label']
        label_all = [mx.nd.array(label)]

        data_batch = SimpleBatch(data_names, data_all, label_names, label_all, cur_bucket_key)
        return data_batch
