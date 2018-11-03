# -*- coding: utf-8 -*-

import copy
import json
import math
import os
import numpy as np
import mxnet as mx

from bucket_io import SimpleBatch
from bucket_io import default_text2id

# The interface of a data iter that works for bucketing
#
# DataIter
#   - default_bucket_key: the bucket key for the default symbol.
#
# DataBatch
#   - provide_data: same as DataIter, but specific to this batch
#   - provide_label: same as DataIter, but specific to this batch
#   - bucket_key: the key for the bucket that should be used for this batch
from mat_data import ontologyDict


def read_nbest_dialog_content(dialog, labelIdx):
    """
    dialog_sentences: (turn_num, nbest_num, sentence_len)
    dialog_scores: (turn_num, nbest_num)
    machine_acts: (turn_num, machine_act_len)
    dialog_labels: (turn_num, )
    """    
    dialog_sentences, dialog_scores, machine_acts, dialog_labels = [], [], [], []
    for turn in dialog["turns"]:
        dialog_labels.append(turn["labelIdx"][labelIdx])

        machine_act = ""
        for saPair in turn["machine_output"]:
            act = saPair["act"]
            slots = " "
            for slot in saPair["slots"]:
                #count never appears in train/dev set#
                if "count" in slot:
                    #slot[1] = str(slot[1])
                    continue
                slots += " ".join(slot)
                slots += " "
            machine_act_item=(act+slots)
            machine_act += machine_act_item
        machine_act = machine_act.strip()
        machine_acts.append(machine_act)

        nbest_sentences = []
        nbest_scores = []
        for asr_hyp in turn["user_input"]:
            if len(asr_hyp["asr-hyp"].split()) == 0:
                continue
            nbest_scores.append(asr_hyp["score"])
            sentence = ""
            #sentence +=" #turn# "
            sentence += asr_hyp["asr-hyp"]
            #sentence += " </s> "
            nbest_sentences.append(sentence)
        dialog_sentences.append(nbest_sentences)
        dialog_scores.append(nbest_scores)

    return dialog_sentences, dialog_scores, machine_acts, dialog_labels

def turn_read_content(path, labelIdx, dataIdx):
    """
    sentences: (dialog_num, turn_num, nbest_num, sentence_len)
    scores: (dialog_num, turn_num, nbest_num)
    acts: (dialog_num, turn_num, machine_act_len)
    labels: (dialog_num, turn_num, [label_dim])
    """
    sentences, scores, acts, labels = [], [], [], []
    with open(path) as json_file:
        data = json.load(json_file)
        #print data["data"][dataIdx]
        for dialog in data[dataIdx]:
            dialog_sentences, dialog_scores, machine_acts, dialog_labels = read_nbest_dialog_content(dialog, labelIdx)
            sentences.append(dialog_sentences)
            scores.append(dialog_scores)
            acts.append(machine_acts)
            labels.append(dialog_labels)
    return sentences, scores, acts, labels

def text2bow(sentence, the_vocab):
    res = np.zeros(len(the_vocab))
    for word in the_vocab:
        if word in sentence:
            res[the_vocab[word]] = 1
    # words = sentence.split()
    # for word in words:
    #     if word in the_vocab:
    #         res[the_vocab[word]] = 1
    return res

class MATTurnSentIter(mx.io.DataIter):
    """
    feature_type,['bowbow', 'sentsent', 'bowsent', 'sentbow']
    """
    def __init__(self, path, labelIdx,vocab, vocab1, buckets, batch_size, max_nbest, max_sentlen,
                 init_states, data_components, label_out=1, feature_type='bowbow'):
        super(MATTurnSentIter, self).__init__()
        self.vocab = vocab
        self.vocab1=vocab1
        self.padding_id = self.vocab['</s>']

        self.label_out = label_out

        self.max_nbest = max_nbest
        self.max_sentlen = max_sentlen
        self.feature_type = feature_type
        self.len_sent = self.max_sentlen if self.feature_type in ['sentsent', 'sentbow'] else len(self.vocab)
        self.len_act_sent = self.max_sentlen if self.feature_type in ['sentsent', 'bowsent'] else len(self.vocab1)

        sentences, scores, acts, labels = turn_read_content(path, labelIdx[0],0)
        #sentences1, scores1, acts1, labels1 = turn_read_content(path, labelIdx[0],1)
        
         
        lab=[]
        for i in labelIdx:
            se,sc,ac,l=turn_read_content(path, i,0)
            lab.append(l)
        
        labels0=[]
        for i in range(len(labels)):
            d0=[]
            for j in range(len(labels[i])):
                ll=[]
                for lb in lab:
                    ll.append(lb[i][j]) 
                d0.append(ll)
            labels0.append(d0)
        labels=labels0


        """
        sentences: (dialog_num, turn_num, nbest_num, sentence_len)
        scores: (dialog_num, turn_num, nbest_num)
        acts: (dialog_num, turn_num, machine_act_len)
        labels: (dialog_num, turn_num, )
        """

        """
        new
        sentences: (dialog_num, turn_num, 2, nbest_num, sentence_len)
        scores: (dialog_num, turn_num, nbest_num)
        acts: (dialog_num, turn_num, 2 , machine_act_len)
        labels: (dialog_num, turn_num,4 )
        """

        buckets.sort()
        self.buckets = buckets
        self.data = [[] for _ in buckets]
        self.data_act = [[] for _ in buckets]
        self.data_score = [[] for _ in buckets]
        self.label = [[] for _ in buckets]

        # pre-allocate with the largest bucket for better memory sharing
        self.default_bucket_key = max(buckets)

        for i in range(len(sentences)):
            sentence = sentences[i]
            score = scores[i]
            act = acts[i]
            label = labels[i]
            for turn_id in range(len(sentence)):
                # user sentence feature
                #for i in range(2):
                for nbest_id in range(len(sentence[turn_id])):        
                    if self.feature_type in ['sentsent', 'sentbow']:
                        sentence[turn_id][nbest_id] = default_text2id(sentence[turn_id][nbest_id], self.vocab)
                    elif self.feature_type in ['bowsent', 'bowbow']:
                        sentence[turn_id][nbest_id] = text2bow(sentence[turn_id][nbest_id], self.vocab)
                # sys act feature
                if self.feature_type in ['sentbow', 'bowbow']:
                    act[turn_id] = text2bow(act[turn_id], self.vocab1)
                elif self.feature_type in ['sentsent', 'bowsent']:
                    act[turn_id] = default_text2id(act[turn_id], self.vocab1)
            for i, bkt in enumerate(buckets): 
                if bkt == len(sentence):
                    self.data[i].append(sentence)
                    self.data_score[i].append(score)
                    self.data_act[i].append(act)
                    self.label[i].append(label)
                    break
            """
            sentence: (turn_num, nbest_num, len_sent)
            score: (turn_num, nbest_num)
            act: (turn_num, len_act_sent)
            label: (turn_num, label_out)
            """
            # we just ignore the sentence it is longer than the maximum
            # bucket size here

        embed_weight = mx.nd.array(np.load('embed_vN3.npy'))
        slotsent="food pricerange name area" 
        slota=default_text2id(slotsent, self.vocab)
        slotarr=slotsent.split()
        #print slota
        label_len=len(labelIdx)
        val_len=[]
        for i in labelIdx:
            val_len.append(len(ontologyDict[u'informable'][slotarr[i]]))

        vl=[]
        for i in labelIdx:
            vla=[]
            for key in ontologyDict[u'informable'][slotarr[i]]:
                #print key
                v=default_text2id(key,self.vocab)
                tmp=mx.nd.array(v)
                tmp= mx.nd.Embedding(data=tmp, input_dim=len(self.vocab), weight=embed_weight, output_dim=300, name='embed')
                tmp=mx.nd.sum(tmp,axis=0)
                v=tmp.asnumpy()
                vla.append(v)
            vla=np.asarray(vla)
            vl.append(vla)
        #vl=np.asarray(vl)
        #print vl
        #print len(vl)
        
        sa=[]
        for i in labelIdx:
            tmp=mx.nd.array([slota[i]])
            tmp= mx.nd.Embedding(data=tmp, input_dim=len(self.vocab), weight=embed_weight, output_dim=300, name='embed')
            sa.append(tmp.asnumpy())
        slota=np.squeeze(np.asarray(sa))
      

        slot=np.zeros((batch_size,label_len,300)) 
        for i in range(batch_size):
                slot[i]=slota                       
        
        value=[]
        for j in range(label_len):
            tmp=np.zeros((batch_size,val_len[j],300)) 
            for i in range(batch_size):
                tmp[i]=vl[j]
            value.append(tmp)
       
        vl_name=[]     
        for i in range(label_len):
            vl_name.append("value_%d"%i)        
        self.vl_name=vl_name

 
        # convert data into ndarrays for better speed during training
        data_mask_len =[np.zeros((len(x), )) for i, x in enumerate(self.data)]
        datatmp =      [np.full((len(x), buckets[i],self.max_nbest,self.len_sent), self.padding_id) for i, x in enumerate(self.data)]
        data_act =  [np.full((len(x), buckets[i],self.len_act_sent), 0.0) for i, x in enumerate(self.data_act)]
        data_score =[np.zeros((len(x), buckets[i], self.max_nbest)) for i, x in enumerate(self.data_score)]
        label =     [np.zeros((len(x), buckets[i], self.label_out)) for i, x in enumerate(self.label)]
        data =[np.zeros((len(x), buckets[i],self.len_sent,300)) for i, x in enumerate(self.data)]
        
        #slot =[np.zeros((len(x), buckets[i],300),dtype=np.float32) for i, x in enumerate(self.data)]
        for i_bucket in range(len(self.buckets)):
            for i_diag in range(len(self.data[i_bucket])):
                data_mask_len[i_bucket][i_diag]=len(self.data[i_bucket][i_diag])
                for i_turn in range(len(self.data[i_bucket][i_diag])):

                    act = self.data_act[i_bucket][i_diag][i_turn]
                    #for i in range(2):
                    data_act[i_bucket][i_diag, i_turn, :len(act)] = act
                    label[i_bucket][i_diag, i_turn, :] = self.label[i_bucket][i_diag][i_turn]
                    # be careful that, here, max_nbest can be smaller than current turn nbest number. extra-best will be truncated.
                    #for i_data in range(2):
                    tempsent=[]    
                    for i_nbest in range(min(len(self.data[i_bucket][i_diag][i_turn]), self.max_nbest)):
                        sentence = self.data[i_bucket][i_diag][i_turn][i_nbest]
                        datatmp[i_bucket][i_diag, i_turn,i_nbest, :len(sentence)] = sentence
                        tmp=mx.nd.array(datatmp[i_bucket][i_diag, i_turn,i_nbest])
                        tmp= mx.nd.Embedding(data=tmp, input_dim=len(self.vocab), weight=embed_weight, output_dim=300, name='embed')
                        sentence=tmp.asnumpy()
                        score = self.data_score[i_bucket][i_diag][i_turn][i_nbest]
                        #preprocess
                        sent =sentence*score 
                        #if i_nbest ==0:
                        tempsent.append(sent)
                        data_score[i_bucket][i_diag, i_turn, i_nbest] = score
                    tempsent=np.asarray(tempsent)
                    scoredsent=np.sum(tempsent,axis=0)
                    #scoredsent=scoredsent*2-1 
                    data[i_bucket][i_diag, i_turn] = scoredsent

        """
        data: (bucket_num, dialog_num, bucket_size/turn_num, max_nbest, len_sent)
        score: (bucket_num, dialog_num, bucket_size/turn_num, max_nbest)
        data_act: (bucket_num, dialog_num, bucket_size/turn_num, len_act_sent)
        label: (bucket_num, dialog_num, bucket_size/turn_num, label_out)
        """
        self.data_mask_len=data_mask_len
        self.data = data
        self.data_act = data_act
        self.data_score = data_score
        self.label = label

        self.slot=slot

        self.value=value

        # backup corpus
        self.all_data_mask_len = copy.deepcopy(self.data_mask_len)
        self.all_data = copy.deepcopy(self.data)
        self.all_data_act = copy.deepcopy(self.data_act)
        self.all_data_score = copy.deepcopy(self.data_score)
        self.all_label = copy.deepcopy(self.label)

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        sizeS=0
        bucket_sizes = [len(x) for x in self.data]
        print("Summary of dataset ==================")
        for bkt, size in zip(buckets, bucket_sizes):
            sizeS+=size
            print("bucket of len %3d : %d samples" % (bkt, size))

        self.batch_size = batch_size
        #self.make_data_iter_plan()

        self.init_states = init_states
        self.data_components = data_components
        self.size=int(sizeS/batch_size)
        self.provide_data = self.data_components + self.init_states


    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        for i in range(len(self.data)):
            # shuffle data before truncate
            index_shuffle = range(len(self.data[i]))
            np.random.shuffle(index_shuffle)
            self.data[i] = self.all_data[i][index_shuffle]
            self.data_mask_len[i] = self.all_data_mask_len[i][index_shuffle]
            self.data_act[i] = self.all_data_act[i][index_shuffle]
            self.data_score[i] = self.all_data_score[i][index_shuffle]
            self.label[i] = self.all_label[i][index_shuffle]

            bucket_n_batches.append(int(math.ceil(1.0*len(self.data[i]) / self.batch_size)))
            self.data[i] = self.data[i][:int(bucket_n_batches[i]*self.batch_size)]
            self.data_mask_len[i] = self.data_mask_len[i][:int(bucket_n_batches[i]*self.batch_size)]
            self.data_act[i] = self.data_act[i][:int(bucket_n_batches[i]*self.batch_size)]
            self.data_score[i] = self.data_score[i][:int(bucket_n_batches[i]*self.batch_size)]
            self.label[i] = self.label[i][:int(bucket_n_batches[i]*self.batch_size)]

        bucket_plan = np.hstack([np.zeros(n, int)+i for i, n in enumerate(bucket_n_batches)])
        np.random.shuffle(bucket_plan)

        bucket_idx_all = [np.random.permutation(len(x)) for x in self.data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        self.bucket_curr_idx = [0 for x in self.data]

        self.data_buffer = []
        self.data_mask_len_buffer = []
        self.data_act_buffer = []
        self.data_score_buffer = []
        self.label_buffer = []
        for i_bucket in range(len(self.data)):
            data = np.zeros((self.batch_size, self.buckets[i_bucket], self.len_sent,300))
            data_mask_len=np.zeros((self.batch_size,))
            data_act = np.zeros((self.batch_size, self.buckets[i_bucket], self.len_act_sent))
            data_score = np.zeros((self.batch_size, self.buckets[i_bucket], self.max_nbest))
            label = np.zeros((self.batch_size, self.buckets[i_bucket], self.label_out))
            self.data_buffer.append(data)
            self.data_mask_len_buffer.append(data_mask_len)
            self.data_act_buffer.append(data_act)
            self.data_score_buffer.append(data_score)
            self.label_buffer.append(label)

    def __iter__(self):
        self.make_data_iter_plan()
        for i_bucket in self.bucket_plan:
            data = self.data_buffer[i_bucket]
            data_mask_len = self.data_mask_len_buffer[i_bucket]
            data_act = self.data_act_buffer[i_bucket]
            data_score = self.data_score_buffer[i_bucket]
            label = self.label_buffer[i_bucket]
            data.fill(self.padding_id)
            data_act.fill(self.padding_id)
            data_score.fill(0)
            label.fill(0)

            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx+self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size

            # Data parallelism
            data[:len(idx)] = self.data[i_bucket][idx]
            data_mask_len[:len(idx)] = self.data_mask_len[i_bucket][idx]
            data_act[:len(idx)] = self.data_act[i_bucket][idx]
            data_score[:len(idx)] = self.data_score[i_bucket][idx]
            label[:len(idx)] = self.label[i_bucket][idx]

            data_names = [x[0] for x in self.provide_data]
            init_state_arrays = [mx.nd.zeros(x[1]) for x in self.init_states]
            data_all = [mx.nd.array(data), mx.nd.array(data_act)]
            if 'score' in data_names:
                data_all += [mx.nd.array(data_score)]
            if 'data_mask_len' in data_names:
                data_all += [mx.nd.array(data_mask_len)]
 
            if 'slot' in data_names:
                data_all += [mx.nd.array(self.slot)]
            
            label_names = ['softmax_label']
            
            label_all = [mx.nd.array(label)]
            #labels=mx.nd.split(mx.nd.array(label),axis=-1,num_outputs=self.label_out,squeeze_axis=1)
            
            #data_batch=[]          
            for i in range(self.label_out):                 
                if self.vl_name[i] in data_names:
                    data_all += [mx.nd.array(self.value[i])]

            data_all += init_state_arrays


            pad = self.batch_size - len(idx)
            
            data_batch=SimpleBatch(data_names, data_all, label_names, label_all, self.buckets[i_bucket], pad)
            yield data_batch

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]
