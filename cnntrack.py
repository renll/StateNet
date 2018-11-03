# -*- coding: utf-8 -*-

import sys,os
import mxnet as mx
import numpy as np
import time
import math
from collections import namedtuple

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # get a logger to accuracies are printed


from data import vocab, ontologyDict as ontology
from bucket_io import default_text2id, default_read_content

# 脚本所在位置
cur_dir = os.path.dirname(os.path.abspath(__file__))


CNNModel = namedtuple("CNNModel", ['cnn_exec', 'symbol', 'data', 'label', 'param_blocks'])

def text_cnn(sentence_size, num_embed, batch_size, vocab_size,
        num_label, filter_list=[3, 4, 5], num_filter=200,
        dropout=0., with_embedding=True):

    input_x = mx.sym.Variable('data') # placeholder for input
    input_y = mx.sym.Variable('softmax_label') # placeholder for output

    # embedding layer
    if not with_embedding:
        embed_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=num_embed, name='vocab_embed')
        conv_input = mx.sym.Reshape(data=embed_layer, target_shape=(batch_size, 1, sentence_size, num_embed))
    else:
        conv_input = input_x

    # create convolution + (max) pooling layer for each filter operation
    pooled_outputs = []
    for i, filter_size in enumerate(filter_list):
        convi = mx.sym.Convolution(data=conv_input, kernel=(filter_size, num_embed), num_filter=num_filter)
        relui = mx.sym.Activation(data=convi, act_type='relu')
        pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(sentence_size - filter_size + 1, 1), stride=(1,1))
        pooled_outputs.append(pooli)

    # combine all pooled outputs
    total_filters = num_filter * len(filter_list)
    concat = mx.sym.Concat(*pooled_outputs, dim=1)
    h_pool = mx.sym.Reshape(data=concat, target_shape=(batch_size, total_filters))

    # dropout layer
    if dropout > 0.0:
        h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
    else:
        h_drop = h_pool

    # fully connected
    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')

    fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias=cls_bias, num_hidden=num_label)

    # softmax output
    #sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax', normalization='batch')
    sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')

    return sm


def setup_cnn_model(ctx, batch_size, sentence_size, num_embed, vocab_size, num_label,
        dropout=0.5, initializer=mx.initializer.Uniform(0.1), with_embedding=True):
        #dropout=0.5, initializer=mx.init.Xavier(magnitude=2.34), with_embedding=True):

    cnn = text_cnn(sentence_size, num_embed, batch_size=batch_size,
            vocab_size=vocab_size, num_label=num_label, dropout=dropout, with_embedding=with_embedding)
    arg_names = cnn.list_arguments()

    input_shapes = {}
    if with_embedding:
        input_shapes['data'] = (batch_size, 1, sentence_size, num_embed)
    else:
        input_shapes['data'] = (batch_size, sentence_size)

    arg_shape, out_shape, aux_shape = cnn.infer_shape(**input_shapes)
    arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
    args_grad = {}
    for shape, name in zip(arg_shape, arg_names):
        if name in ['softmax_label', 'data']: # input, output
            continue
        args_grad[name] = mx.nd.zeros(shape, ctx)

    cnn_exec = cnn.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')

    param_blocks = []
    arg_dict = dict(zip(arg_names, cnn_exec.arg_arrays))
    for i, name in enumerate(arg_names):
        if name in ['softmax_label', 'data']: # input, output
            continue
        initializer(name, arg_dict[name])

        param_blocks.append( (i, arg_dict[name], args_grad[name], name) )

    out_dict = dict(zip(cnn.list_outputs(), cnn_exec.outputs))

    data = cnn_exec.arg_dict['data']
    label = cnn_exec.arg_dict['softmax_label']

    return CNNModel(cnn_exec=cnn_exec, symbol=cnn, data=data, label=label, param_blocks=param_blocks)


def train_cnn(model, X_train_batch, y_train_batch, X_dev_batch, y_dev_batch, X_test_batch, y_test_batch, batch_size,
        #optimizer='rmsprop', max_grad_norm=5.0, learning_rate=0.0005, epoch=200):
        #optimizer='adadelta', max_grad_norm=5.0, learning_rate=0.0005, epoch=200):
        optimizer='adam', max_grad_norm=5.0, learning_rate=0.0005, epoch=100):
    m = model
    # create optimizer
    opt = mx.optimizer.create(optimizer)
    opt.lr = learning_rate

    updater = mx.optimizer.get_updater(opt)

    dev_acc_list = [0.0]
    for iteration in range(epoch):
        tic = time.time()
        num_correct = 0
        num_total = 0
        for begin in range(0, X_train_batch.shape[0], batch_size):
            batchX = X_train_batch[begin:begin+batch_size]
            batchY = y_train_batch[begin:begin+batch_size]
            if batchX.shape[0] != batch_size:
                continue

            m.data[:] = batchX
            m.label[:] = batchY

            # forward
            m.cnn_exec.forward(is_train=True)

            # backward
            m.cnn_exec.backward()

            # eval on training data
            num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

            # update weights
            norm = 0
            for idx, weight, grad, name in m.param_blocks:
                grad /= batch_size
                l2_norm = mx.nd.norm(grad).asscalar()
                norm += l2_norm * l2_norm

            norm = math.sqrt(norm)
            for idx, weight, grad, name in m.param_blocks:
                if norm > max_grad_norm:
                    grad *= (max_grad_norm / norm)

                updater(idx, grad, weight)

                # reset gradient to zero
                grad[:] = 0.0

        # decay learning rate
        #if iteration % 50 == 0 and iteration > 0:
        #    opt.lr *= 0.5
        #    print('reset learning rate to %g' % opt.lr)

        # end of training loop
        toc = time.time()
        train_time = toc - tic
        train_acc = num_correct * 100 / float(num_total)

        # saving checkpoint
        if (iteration + 1) % 10 == 0:
            prefix = 'cnn'
            m.symbol.save('checkpoint/%s-symbol.json' % prefix)
            save_dict = {('arg:%s' % k) :v  for k, v in m.cnn_exec.arg_dict.items()}
            save_dict.update({('aux:%s' % k) : v for k, v in m.cnn_exec.aux_dict.items()})
            param_name = 'checkpoint/%s-%04d.params' % (prefix, iteration)
            mx.nd.save(param_name, save_dict)
            print('Saved checkpoint to %s' % param_name)

        def evaluate_dataset(X_batch, y_batch):
            # evaluate on some data set
            num_correct = 0
            num_total = 0
            for begin in range(0, X_batch.shape[0], batch_size):
                batchX = X_batch[begin:begin+batch_size]
                batchY = y_batch[begin:begin+batch_size]
                if batchX.shape[0] != batch_size:
                    continue

                m.data[:] = batchX
                m.cnn_exec.forward(is_train=False)
                num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
                num_total += len(batchY)
            acc = num_correct * 100 / float(num_total)
            return acc

        dev_acc = evaluate_dataset(X_dev_batch, y_dev_batch)
        test_acc = evaluate_dataset(X_test_batch, y_test_batch)
        print('Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f \
                --- Dev Accuracy thus far: %.3f \
                --- Test Accuracy thus far: %.3f' % (iteration, train_time, train_acc, dev_acc, test_acc))
        sys.stdout.flush()
        sys.stderr.flush()

        # decay learning rate
        #if dev_acc < dev_acc_list[-1]:
        #    opt.lr *= 0.5
        #    print('reset learning rate to %g' % opt.lr)
        #dev_acc_list.append(dev_acc)



def get_x_y_from_data(data_json_file, labelIdx):
    raw_sentences, scores, labels = default_read_content(data_json_file, labelIdx)
    sentences = []
    for i in xrange(len(raw_sentences)):
        raw_sentence = raw_sentences[i]
        sentences.append(default_text2id(raw_sentence, vocab))

    # padding to max sentence length with '</s>'
    padding_word = '</s>'
    sequence_length = 360
    padded_sentences = []
    for i in xrange(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [vocab[padding_word]] * num_padding
        padded_sentences.append(new_sentence)

    # convert to np array
    x = np.array(padded_sentences)
    y = np.array(labels)
    return x, y


def train_without_pretrained_embedding(labelIdx, config_dict={}):
    # ##################################
    label_index_list = ['goal_food', 'goal_pricerange', 'goal_name', 'goal_area', 'method', 'requested']
    num_label_dict = {
        'goal_food':        len(ontology["informable"]["food"])+1,
        'goal_pricerange':  len(ontology["informable"]["pricerange"])+1,
        'goal_name':        len(ontology["informable"]["name"])+1,
        'goal_area':        len(ontology["informable"]["area"])+1,
        'method':           len(ontology["method"]),
        'requested':        len(ontology["requestable"])
    }
    vocab_size = len(vocab)
    np.random.seed(10)

    # ##################################
    train_json_file    = config_dict.get('train_json', os.path.join(cur_dir, 'train_nbest.json'))
    dev_json_file      = config_dict.get('dev_json',   os.path.join(cur_dir, 'dev_nbest.json'))
    test_json_file     = config_dict.get('test_json',  os.path.join(cur_dir, 'test_nbest.json'))

    label_name = label_index_list[labelIdx]
    num_label = num_label_dict[label_name]
    x_train, y_train = get_x_y_from_data(train_json_file, labelIdx)
    x_dev, y_dev = get_x_y_from_data(dev_json_file, labelIdx)
    x_test, y_test = get_x_y_from_data(test_json_file, labelIdx)

    # ##################################
    # randomly shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    shuffle_indices = np.random.permutation(np.arange(len(y_dev)))
    x_dev = x_dev[shuffle_indices]
    y_dev = y_dev[shuffle_indices]
    shuffle_indices = np.random.permutation(np.arange(len(y_test)))
    x_test = x_test[shuffle_indices]
    y_test = y_test[shuffle_indices]
    print('Train/Dev split: %d/%d' % (len(y_train), len(y_dev)))
    print('train shape:', x_train.shape)
    print('dev shape:', x_dev.shape)
    print('test shape:', x_test.shape)
    print('vocab_size', vocab_size)

    batch_size = 32
    num_embed = 100
    sentence_size = x_train.shape[1]

    print('batch size', batch_size)
    print('sentence max words', sentence_size)
    print('embedding size', num_embed)

    cnn_model = setup_cnn_model(mx.gpu(0), batch_size, sentence_size, num_embed, vocab_size, num_label, dropout=0.5, with_embedding=False)
    train_cnn(cnn_model, x_train, y_train, x_dev, y_dev, x_test, y_test, batch_size)


#class CnnTrack(object):
#    """CnnTrack implementation, config_dict:
#        output_type: can be one of ['softmax', 'sigmoid']
#        N: number of cpu/gpu cores
#        pretrain_embed: whether to use pre-trained word embedding
#        embed_matrix: pre-trained embed_matrix file
#    """
#    def __init__(self, config_dict):
#        pass


if __name__ == '__main__':
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    train_without_pretrained_embedding(0)
    #for i in xrange(5):
    #    print('labelIdx: ', i)
    #    train_without_pretrained_embedding(i)


