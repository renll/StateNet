# -*- coding: utf-8 -*-

import copy
import json
import math

import numpy as np
import mxnet as mx

from bucket_io import SimpleBatch
from turnsent_io import turn_read_content, text2bow

# The interface of a data iter that works for bucketing
#
# DataIter
#   - default_bucket_key: the bucket key for the default symbol.
#
# DataBatch
#   - provide_data: same as DataIter, but specific to this batch
#   - provide_label: same as DataIter, but specific to this batch
#   - bucket_key: the key for the bucket that should be used for this batch

def nbest_text2bow(nbest_sentence, nbest_score, the_vocab, ngram=1):
    res = np.zeros(len(the_vocab))
    for i in range(len(nbest_sentence)):
        words = list(set(nbest_sentence[i].split()))
        for word in words:
            if word in the_vocab:
                res[the_vocab[word]] += nbest_score[i]
        if ngram >= 2:
            for word in [' '.join(words[j:j+2]) for j in xrange(len(words)-1)]:
                if word in the_vocab:
                    res[the_vocab[word]] += nbest_score[i]
        if ngram >= 3:
            for word in [' '.join(words[j:j+3]) for j in xrange(len(words)-2)]:
                if word in the_vocab:
                    res[the_vocab[word]] += nbest_score[i]
    return res

class DSTTurnIter(mx.io.DataIter):
    def __init__(self, path, labelIdx, vocab, buckets, batch_size,
                 init_states, data_components, label_out=1):
        super(DSTTurnIter, self).__init__()
        self.vocab = vocab
        self.padding_id = self.vocab['</s>']

        self.label_out = label_out

        sentences, scores, acts, labels = turn_read_content(path, labelIdx)
        """
        sentences: (dialog_num, turn_num, nbest_num, sentence_len)
        scores: (dialog_num, turn_num, nbest_num)
        acts: (dialog_num, turn_num, machine_act_len)
        labels: (dialog_num, turn_num, )
        """

        buckets.sort()
        self.buckets = buckets
        self.data = [[] for _ in buckets]
        self.data_act = [[] for _ in buckets]
        self.label = [[] for _ in buckets]

        # pre-allocate with the largest bucket for better memory sharing
        self.default_bucket_key = max(buckets)

        for i in range(len(sentences)):
            sentence = sentences[i]
            score = scores[i]
            act = acts[i]
            label = labels[i]
            for turn_id in range(len(sentence)):
                sentence[turn_id] = nbest_text2bow(sentence[turn_id], score[turn_id], self.vocab, ngram=1)
                act[turn_id] = text2bow(act[turn_id], self.vocab)
            for i, bkt in enumerate(buckets):
                if bkt == len(sentence):
                    self.data[i].append(sentence)
                    self.data_act[i].append(act)
                    self.label[i].append(label)
                    break
            """
            sentence: (turn_num, vocab_size)
            act: (turn_num, vocab_size)
            label: (turn_num, label_out)
            """
            # we just ignore the sentence it is longer than the maximum
            # bucket size here

        # convert data into ndarrays for better speed during training
        data = [np.array(x) for i, x in enumerate(self.data)]
        data_act = [np.array(x) for i, x in enumerate(self.data_act)]
        label = [np.array(x).reshape((len(x), buckets[i], self.label_out)) for i, x in enumerate(self.label)]

        self.data = data
        self.data_act = data_act
        self.label = label

        # backup corpus
        self.all_data = copy.deepcopy(self.data)
        self.all_data_act = copy.deepcopy(self.data_act)
        self.all_label = copy.deepcopy(self.label)

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.data]
        print("Summary of dataset ==================")
        for bkt, size in zip(buckets, bucket_sizes):
            print("bucket of len %3d : %d samples" % (bkt, size))

        self.batch_size = batch_size
        #self.make_data_iter_plan()

        self.init_states = init_states
        self.data_components = data_components

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
            self.data_act[i] = self.all_data_act[i][index_shuffle]
            self.label[i] = self.all_label[i][index_shuffle]

            bucket_n_batches.append(int(math.ceil(1.0*len(self.data[i]) / self.batch_size)))
            self.data[i] = self.data[i][:int(bucket_n_batches[i]*self.batch_size)]
            self.data_act[i] = self.data_act[i][:int(bucket_n_batches[i]*self.batch_size)]
            self.label[i] = self.label[i][:int(bucket_n_batches[i]*self.batch_size)]

        bucket_plan = np.hstack([np.zeros(n, int)+i for i, n in enumerate(bucket_n_batches)])
        np.random.shuffle(bucket_plan)

        bucket_idx_all = [np.random.permutation(len(x)) for x in self.data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        self.bucket_curr_idx = [0 for x in self.data]

        self.data_buffer = []
        self.data_act_buffer = []
        self.label_buffer = []
        for i_bucket in range(len(self.data)):
            data = np.zeros((self.batch_size, self.buckets[i_bucket], len(self.vocab)))
            data_act = np.zeros((self.batch_size, self.buckets[i_bucket], len(self.vocab)))
            label = np.zeros((self.batch_size, self.buckets[i_bucket], self.label_out))
            self.data_buffer.append(data)
            self.data_act_buffer.append(data_act)
            self.label_buffer.append(label)

    def __iter__(self):
        self.make_data_iter_plan()
        for i_bucket in self.bucket_plan:
            data = self.data_buffer[i_bucket]
            data_act = self.data_act_buffer[i_bucket]
            label = self.label_buffer[i_bucket]
            data.fill(0)
            data_act.fill(0)
            label.fill(0)

            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx+self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size

            # Data parallelism
            data[:len(idx)] = self.data[i_bucket][idx]
            data_act[:len(idx)] = self.data_act[i_bucket][idx]
            label[:len(idx)] = self.label[i_bucket][idx]

            data_names = [x[0] for x in self.provide_data]
            init_state_arrays = [mx.nd.zeros(x[1]) for x in self.init_states]
            data_all = [mx.nd.array(data), mx.nd.array(data_act)]
            data_all += init_state_arrays

            label_names = ['softmax_label']
            label_all = [mx.nd.array(label)]

            pad = self.batch_size - len(idx)
            data_batch = SimpleBatch(data_names, data_all, label_names, label_all, self.buckets[i_bucket], pad)
            yield data_batch

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]
