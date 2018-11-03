# -*- coding: utf-8 -*-

import copy
import json

import numpy as np
import mxnet as mx

# The interface of a data iter that works for bucketing
#
# DataIter
#   - default_bucket_key: the bucket key for the default symbol.
#
# DataBatch
#   - provide_data: same as DataIter, but specific to this batch
#   - provide_label: same as DataIter, but specific to this batch
#   - bucket_key: the key for the bucket that should be used for this batch

def read_1best_dialog_content(dialog, labelIdx):
    dialog_sentences, dialog_scores, dialog_labels = [], [], []
    sentence = ""
    score = []
    for turn in dialog["turns"]:
        dialog_labels.append(turn["labelIdx"][labelIdx])
        sentence +=" #turn# "
        score.append(1)

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
            machine_act=(act+slots)
            for _ in range(len(machine_act.split())):
                score.append(1)
            sentence += machine_act

        asr = turn["user_input"][0]["asr-hyp"]
        if len(asr.split()) > 0:
            sentence += turn["user_input"][0]["asr-hyp"] + " "
            score.extend([turn["user_input"][0]["score"]] * len(asr.split()))

        #sentence += " </s> "
        #score.append(1)
        assert(len(sentence.split())==len(score))
        dialog_sentences.append(sentence)
        dialog_scores.append(score[:])
    return dialog_sentences, dialog_scores, dialog_labels

def default_read_content(path, labelIdx):
    sentences, scores, labels = [], [], []
    with open(path) as json_file:
        data = json.load(json_file)
        for dialog in data:
            dialog_sentences, dialog_scores, dialog_labels = read_1best_dialog_content(dialog, labelIdx)
            sentences.extend(dialog_sentences)
            scores.extend(dialog_scores)
            labels.extend(dialog_labels)
    return sentences, scores, labels

def default_text2id(sentence, the_vocab):
    words = sentence.split()
    words = [(the_vocab[w] if w in the_vocab else 0)  for w in words if len(w) > 0]
    return words

def default_gen_buckets(sentences, batch_size, the_vocab):
    len_dict = {}
    max_len = -1
    for sentence in sentences:
        words = default_text2id(sentence, the_vocab)
        if len(words) == 0:
            continue
        if len(words) > max_len:
            max_len = len(words)
        if len(words) in len_dict:
            len_dict[len(words)] += 1
        else:
            len_dict[len(words)] = 1
    #print(len_dict)

    tl = 0
    buckets = []
    for l, n in len_dict.items(): # TODO: There are better heuristic ways to do this
        if n + tl >= batch_size*6:
            buckets.append(l)
            tl = 0
        else:
            tl += n
    if tl > 0 and len(buckets) > 0:
        buckets[-1] = max_len
    return buckets

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key, pad=0):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = pad
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class DSTSentenceIter(mx.io.DataIter):
    def __init__(self, path, labelIdx, vocab, buckets, batch_size,
                 init_states, data_components,
                 seperate_char=' <eos> ', text2id=None, read_content=None, label_out=1):
        super(DSTSentenceIter, self).__init__()
        self.padding_id = vocab['</s>']

        self.label_out = label_out
        if text2id == None:
            self.text2id = default_text2id
        else:
            self.text2id = text2id
        if read_content == None:
            self.read_content = default_read_content
        else:
            self.read_content = read_content
        #content = self.read_content(path)
        sentences,scores,labels = self.read_content(path, labelIdx)

        if len(buckets) == 0:
            buckets = default_gen_buckets(sentences, batch_size, vocab)

        self.vocab_size = len(vocab)

        buckets.sort()
        self.buckets = buckets
        self.data = [[] for _ in buckets]
        self.data_score = [[] for _ in buckets]
        self.label = [[] for _ in buckets]

        # pre-allocate with the largest bucket for better memory sharing
        self.default_bucket_key = max(buckets)

        for i in range(len(sentences)):
            sentence = sentences[i]
            score = scores[i]
            label = labels[i]
            sentence = self.text2id(sentence, vocab)
            if len(sentence) == 0:
                continue
            for i, bkt in enumerate(buckets):
                if bkt >= len(sentence):
                    assert(len(sentence)==len(score))
                    self.data[i].append(sentence)
                    self.data_score[i].append(score)
                    self.label[i].append(label)
                    break
            # we just ignore the sentence it is longer than the maximum
            # bucket size here

        # re-arrange buckets to include as much as possible corpus
        for i in xrange(len(self.data)-1):
            tmp_num = len(self.data[i]) / batch_size
            self.data[i+1].extend(self.data[i][tmp_num*batch_size:])
            self.data[i] = self.data[i][:tmp_num*batch_size]
            self.data_score[i+1].extend(self.data_score[i][tmp_num*batch_size:])
            self.data_score[i] = self.data_score[i][:tmp_num*batch_size]
            self.label[i+1].extend(self.label[i][tmp_num*batch_size:])
            self.label[i] = self.label[i][:tmp_num*batch_size]

        # convert data into ndarrays for better speed during training
        #data = [np.zeros((len(x), buckets[i])) for i, x in enumerate(self.data)]
        data = [np.full((len(x), buckets[i]), self.padding_id) for i, x in enumerate(self.data)]
        data_mask_len = [np.zeros((len(x), )) for i, x in enumerate(self.data)]
        data_score = [np.zeros((len(x), buckets[i])) for i, x in enumerate(self.data_score)]
        label = [np.zeros((len(x), self.label_out)) for i, x in enumerate(self.label)]
        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.data[i_bucket])):
                sentence = self.data[i_bucket][j]
                data[i_bucket][j, :len(sentence)] = sentence
                data_mask_len[i_bucket][j] = len(sentence)
                score = self.data_score[i_bucket][j]
                #print(sentence)
                #print(score)
                data_score[i_bucket][j, :len(score)] = score
                label[i_bucket][j] = self.label[i_bucket][j]

        self.data = data
        self.data_mask_len = data_mask_len
        self.data_score = data_score
        self.label = label

        # backup corpus
        self.all_data = copy.deepcopy(self.data)
        self.all_data_mask_len = copy.deepcopy(self.data_mask_len)
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
        self.provide_label = [('softmax_label', (self.batch_size, self.label_out))]


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
            self.data_score[i] = self.all_data_score[i][index_shuffle]
            self.label[i] = self.all_label[i][index_shuffle]

            bucket_n_batches.append(len(self.data[i]) / self.batch_size)
            self.data[i] = self.data[i][:int(bucket_n_batches[i]*self.batch_size)]
            self.data_mask_len[i] = self.data_mask_len[i][:int(bucket_n_batches[i]*self.batch_size)]
            self.data_score[i] = self.data_score[i][:int(bucket_n_batches[i]*self.batch_size)]

        bucket_plan = np.hstack([np.zeros(n, int)+i for i, n in enumerate(bucket_n_batches)])
        np.random.shuffle(bucket_plan)

        bucket_idx_all = [np.random.permutation(len(x)) for x in self.data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        self.bucket_curr_idx = [0 for x in self.data]

        self.data_buffer = []
        self.data_mask_len_buffer = []
        self.data_score_buffer = []
        self.label_buffer = []
        for i_bucket in range(len(self.data)):
            data = np.zeros((self.batch_size, self.buckets[i_bucket]))
            data_mask_len = np.zeros((self.batch_size,))
            data_score = np.zeros((self.batch_size, self.buckets[i_bucket]))
            label = np.zeros((self.batch_size, self.label_out))
            self.data_buffer.append(data)
            self.data_mask_len_buffer.append(data_mask_len)
            self.data_score_buffer.append(data_score)
            self.label_buffer.append(label)

    def __iter__(self):
        self.make_data_iter_plan()
        for i_bucket in self.bucket_plan:
            data = self.data_buffer[i_bucket]
            data_mask_len = self.data_mask_len_buffer[i_bucket]
            data_score = self.data_score_buffer[i_bucket]
            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx+self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size

            # Data parallelism
            data[:] = self.data[i_bucket][idx]
            data_mask_len[:] = self.data_mask_len[i_bucket][idx]
            data_score[:] = self.data_score[i_bucket][idx]

            for sentence in data:
                assert len(sentence) == self.buckets[i_bucket]

            label = self.label_buffer[i_bucket]
            label[:] = self.label[i_bucket][idx]

            data_names = [x[0] for x in self.provide_data]
            init_state_arrays = [mx.nd.zeros(x[1]) for x in self.init_states]
            data_all = [mx.nd.array(data)]
            if 'score' in data_names:
                data_all += [mx.nd.array(data_score)]
            if 'data_mask_len' in data_names:
                data_all += [mx.nd.array(data_mask_len)]
            data_all += init_state_arrays

            label_names = ['softmax_label']
            label_all = [mx.nd.array(label)]

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all, self.buckets[i_bucket])
            yield data_batch


    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]
