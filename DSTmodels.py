# pylint:skip-file
import mxnet as mx
import numpy as np
from collections import namedtuple
import copy

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
MATParam = namedtuple("MATParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias",
                                     "i2g_weight", "i2g_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])


def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


# we define a new unrolling function here because the original
# one in lstm.py Concats all the labels at the last layer together,
# making the mini-batch size of the label different from the data.
# I think the existing data-parallelization code need some modification
# to allow this situation to work properly
def lstm_unroll(num_lstm_layer, seq_len, input_size,
                num_hidden, num_embed,num_lstm_o, num_label, output_type='softmax', dropout=0., fix_embed=False, enable_mask=False):

    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    fc_weight = mx.sym.Variable("fc_weight")
    fc_bias= mx.sym.Variable("fc_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    # embeding layer
    data = mx.sym.Variable('data')
    score = mx.sym.Variable('score')
    score_exp = mx.sym.expand_dims(score,axis=2)
    label = mx.sym.Variable('softmax_label')
    embed = mx.sym.Embedding(data=data, input_dim=input_size, weight=embed_weight, output_dim=num_embed, name='embed')
    if fix_embed:
        embed = mx.sym.BlockGrad(embed)

    Concat_input = mx.sym.Concat(embed,score_exp,dim=2,name='Concat')

    if enable_mask:
        data_mask_len = mx.sym.Variable('data_mask_len')
        Concat_input = mx.sym.SwapAxis(Concat_input, dim1=0, dim2=1)
        Concat_input = mx.sym.SequenceMask(data=Concat_input, use_sequence_length=True, sequence_length=data_mask_len, value=0.)
        Concat_input = mx.sym.SwapAxis(Concat_input, dim1=0, dim2=1)

    lstm_input = mx.sym.SliceChannel(data=Concat_input, num_outputs=seq_len, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len):


        hidden = lstm_input[seqidx]
        hidden = mx.sym.FullyConnected(data=hidden,num_hidden = num_hidden,weight=fc_weight,bias=fc_bias,name='FC_%d'%seqidx)
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)

        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_lstm_o, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    if enable_mask:
        hidden_Concat = mx.sym.Concat(*hidden_all, dim=0)
        pred_all = mx.sym.FullyConnected(data=hidden_Concat, num_hidden=num_label, weight=cls_weight, bias=cls_bias, name='pred')
        pred_all = mx.sym.Reshape(data=pred_all, shape=(seq_len, -1, num_label))
        pred = mx.sym.SequenceLast(data=pred_all, sequence_length=data_mask_len, use_sequence_length=True)
    else:
        hidden_final = hidden_all[-1]
        pred = mx.sym.FullyConnected(data=hidden_final, num_hidden=num_label, weight=cls_weight, bias=cls_bias, name='pred')

    if output_type == 'sigmoid':
        sm = mx.sym.LogisticRegressionOutput(data=pred, label=label, name='softmax')
    else:
        label = mx.sym.Reshape(data=label, shape=(-1,))
        sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    if enable_mask:
        provide_data = ('data', 'score', 'data_mask_len', )
    else:
        provide_data = ('data', 'score', )
    init_states = ['l%d_init_c'%l for l in range(num_lstm_layer)]
    init_states += ['l%d_init_h'%l for l in range(num_lstm_layer)]

    return sm, provide_data+tuple(init_states), ('softmax_label',)

def BAKlstm_unroll(num_lstm_layer, seq_len, input_size,
                num_hidden, num_embed,num_lstm_o, num_label, output_type='softmax', dropout=0., fix_embed=False, enable_mask=False):

    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    fc_weight = mx.sym.Variable("fc_weight")
    fc_bias= mx.sym.Variable("fc_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    # embeding layer
    data = mx.sym.Variable('data')
    score = mx.sym.Variable('score')
    score_exp = mx.sym.expand_dims(score,axis=2)
    label = mx.sym.Variable('softmax_label')
    embed = mx.sym.Embedding(data=data, input_dim=input_size, weight=embed_weight, output_dim=num_embed, name='embed')
    if fix_embed:
        embed = mx.sym.BlockGrad(embed)
    Concat_input = mx.sym.Concat(embed,score_exp,dim=2,name='Concat')

    if enable_mask:
        data_mask_len = mx.sym.Variable('data_mask_len')
        Concat_input = mx.sym.SwapAxis(Concat_input, dim1=0, dim2=1)
        Concat_input = mx.sym.SequenceMask(data=Concat_input, use_sequence_length=True, sequence_length=data_mask_len, value=0.)
        Concat_input = mx.sym.SwapAxis(Concat_input, dim1=0, dim2=1)

    lstm_input = mx.sym.SliceChannel(data=Concat_input, num_outputs=seq_len, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len):


        hidden = lstm_input[seqidx]
        hidden = mx.sym.FullyConnected(data=hidden,num_hidden = num_hidden,weight=fc_weight,bias=fc_bias,name='FC_%d'%seqidx)
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)

        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_lstm_o, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)



    if enable_mask:
        hidden_Concat = mx.sym.Concat(*hidden_all, dim=0)
        pred_all = mx.sym.FullyConnected(data=hidden_Concat, num_hidden=num_label, weight=cls_weight, bias=cls_bias, name='pred')
        pred_all = mx.sym.Reshape(data=pred_all, shape=(seq_len, -1, num_label))
        pred = mx.sym.SequenceLast(data=pred_all, sequence_length=data_mask_len, use_sequence_length=True)
    else:
        hidden_final = hidden_all[-1]
        pred = mx.sym.FullyConnected(data=hidden_final, num_hidden=num_label, weight=cls_weight, bias=cls_bias, name='pred')

    if output_type == 'sigmoid':
        sm = mx.sym.LogisticRegressionOutput(data=pred, label=label, name='softmax')
    else:
        label = mx.sym.Reshape(data=label, shape=(-1,))
        sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    if enable_mask:
        provide_data = ('data', 'score', 'data_mask_len', )
    else:
        provide_data = ('data', 'score', )
    init_states = ['l%d_init_c'%l for l in range(num_lstm_layer)]
    init_states += ['l%d_init_h'%l for l in range(num_lstm_layer)]

    return sm, provide_data+tuple(init_states), ('softmax_label',)

def lstmcnn_unroll(num_lstm_layer, seq_len, input_size,
                num_hidden, num_embed,num_lstm_o, num_label, output_type='softmax', dropout=0., fix_embed=False, enable_mask=False):

    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    fc_weight = mx.sym.Variable("fc_weight")
    fc_bias= mx.sym.Variable("fc_bias")
    last_state = LSTMState(c=mx.sym.Variable("l%d_init_c" % 0),
                          h=mx.sym.Variable("l%d_init_h" % 0))
    data = mx.sym.Variable('data')
    #score = mx.sym.Variable('score')
    score_exp = mx.sym.expand_dims(score,axis=2)
    #label = mx.sym.Variable('softmax_label')
    embed = mx.sym.Embedding(data=data, input_dim=input_size, weight=embed_weight, output_dim=num_embed, name='embed')
    if fix_embed:
        embed = mx.sym.BlockGrad(embed)
    #Concat_input = mx.sym.Concat(embed,score_exp,dim=2,name='Concat')

    lstm_input = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=0)
    
    label = mx.sym.Variable('softmax_label')
    conv_weight = mx.sym.Variable("conv_weight")
    conv_bias = mx.sym.Variable("conv_bias")
    conv1_weight = mx.sym.Variable("conv1_weight")
    conv1_bias = mx.sym.Variable("conv1_bias")
    ffc_weight = mx.sym.Variable("ffc_weight")
    ffc_bias = mx.sym.Variable("ffc_bias")
    conv2_weight = mx.sym.Variable("conv2_weight")
    conv2_bias = mx.sym.Variable("conv2_bias")
    conv3_weight = mx.sym.Variable("conv3_weight")
    conv3_bias = mx.sym.Variable("conv3_bias")
    
 
    g1=mx.sym.Variable("bn1_gamma")
    b1=mx.sym.Variable("bn1_beta")
    mv1=mx.sym.Variable("bn1_moving_var")
    mm1=mx.sym.Variable("bn1_moving_mean")
    g2=mx.sym.Variable("bn2_gamma")
    b2=mx.sym.Variable("bn2_beta")
    mm2=mx.sym.Variable("bn2_moving_mean")
    mv2=mx.sym.Variable("bn2_moving_var")
    g3=mx.sym.Variable("bn3_gamma")
    b3=mx.sym.Variable("bn3_beta")
    mm3=mx.sym.Variable("bn3_moving_mean")
    mv3=mx.sym.Variable("bn3_moving_var")
    g4=mx.sym.Variable("bn4_gamma")
    b4=mx.sym.Variable("bn4_beta")
    mm4=mx.sym.Variable("bn4_moving_mean")
    mv4=mx.sym.Variable("bn4_moving_var")
 
    #label = mx.sym.Variable('softmax_label') # (batch_size, seq_len, label_out)
    nf=16
    hidden_all = []
    for seqidx in range(seq_len):

        x = lstm_input[seqidx]#(b,1,em)
        #temp = mx.sym.SliceChannel(data=x,axis=1, num_outputs=2, squeeze_axis=0)
        #f=temp[0]
        #fs=temp[1]
 
        mem=mx.sym.Reshape(last_state.c,shape=(0,1,num_embed))
        m_in= mx.sym.Concat(f,mem, dim=1) 
        
        x= mx.sym.Concat(x,mem, dim=1)
        
        #x= mx.sym.Concat(x,last_state.h, dim=1)
        mo= mx.sym.FullyConnected(data=m_in,num_hidden = num_embed,weight=ffc_weight,bias=ffc_bias,name='FC_1')
        next_c= mx.sym.Activation(mo, act_type="sigmoid")
        #x=mx.sym.Reshape(x,shape=(0,3,616))
        x= mx.sym.Convolution(data=x, kernel=(15,),stride=(1,),pad=(7,), num_filter=nf,
                                weight=conv_weight, bias=conv_bias,name='conv')
        #x= mx.sym.BatchNorm(data=x,fix_gamma=False)
        x=mx.sym.BatchNorm(data=x,fix_gamma=False,momentum=0.9,gamma=g1,beta=b1,moving_var=mv1,moving_mean=mm1,name='bn1')
        
        x0= mx.sym.Activation(data=x, act_type="relu")
        x= mx.sym.Convolution(data=x0, kernel=(15,), stride=(1,),num_filter=nf, pad=(7,),
                                    weight=conv1_weight, bias=conv1_bias,name='conv1')
        
        x=mx.sym.BatchNorm(data=x,fix_gamma=False,momentum=0.9,gamma=g2,beta=b2,moving_var=mv2,moving_mean=mm2,name='bn2')
        #x= mx.sym.BatchNorm(x)
        x = mx.sym.Activation(data=x, act_type="relu")
        x= mx.sym.Convolution(data=x, kernel=(15,), stride=(1,),num_filter=nf, pad=(7,),
                                    weight=conv3_weight, bias=conv3_bias,name='conv3')
        x=mx.sym.BatchNorm(data=x,fix_gamma=False,momentum=0.9,gamma=g3,beta=b3,moving_var=mv3,moving_mean=mm3,name='bn3')
        #x= mx.sym.BatchNorm(x)
        x = x0+x
        x = mx.sym.Activation(data=x, act_type="relu")
        x= mx.sym.Convolution(data=x, kernel=(1,), stride=(1,),num_filter=1,
                                    weight=conv2_weight, bias=conv2_bias,name='conv2')
        x=mx.sym.BatchNorm(data=x,fix_gamma=False,momentum=0.9,gamma=g4,beta=b4,moving_var=mv4,moving_mean=mm4,name='bn4')
        #x= mx.sym.BatchNorm(x)
        next_h= mx.sym.Activation(data=x, act_type="relu")
        last_state=LSTMState(c=next_c, h=next_h)
        hidden = next_h
        hidden_all.append(hidden)
    
    x=hidden_all[-1]        
    pred = mx.sym.FullyConnected(data=x, num_hidden=num_label, weight=cls_weight, bias=cls_bias, name='pred')

    if output_type == 'sigmoid':
        sm = mx.sym.LogisticRegressionOutput(data=pred, label=label, name='softmax')
    else:
        label = mx.sym.Reshape(data=label, shape=(-1,))
        sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    provide_data = ('data', )
    init_states = ['l%d_init_c'%l for l in range(num_lstm_layer)]
    init_states += ['l%d_init_h'%l for l in range(num_lstm_layer)]

    return sm, provide_data+tuple(init_states), ('softmax_label',)



    

# define blstm
def blstm_unroll(num_lstm_layer, seq_len, input_size,
                num_hidden, num_embed,num_lstm_o, num_label, output_type='softmax', dropout=0., fix_embed=False, enable_mask=False):

    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    fc_weight = mx.sym.Variable("fc_weight")
    fc_bias= mx.sym.Variable("fc_bias")

    forward_param_cells = []
    backward_param_cells = []
    forward_last_states = []
    backward_last_states = []
    for i in range(num_lstm_layer):
        # forward
        prefix = "forward_"
        forward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable(prefix+"l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable(prefix+"l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable(prefix+"l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable(prefix+"l%d_h2h_bias" % i)))
        forward_state = LSTMState(c=mx.sym.Variable(prefix+"l%d_init_c" % i),
                          h=mx.sym.Variable(prefix+"l%d_init_h" % i))
        forward_last_states.append(forward_state)
        # backward
        prefix = "backward_"
        backward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable(prefix+"l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable(prefix+"l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable(prefix+"l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable(prefix+"l%d_h2h_bias" % i)))
        backward_state = LSTMState(c=mx.sym.Variable(prefix+"l%d_init_c" % i),
                          h=mx.sym.Variable(prefix+"l%d_init_h" % i))
        backward_last_states.append(backward_state)
    assert(len(forward_last_states) == num_lstm_layer)
    assert(len(backward_last_states) == num_lstm_layer)

    # embeding layer
    data = mx.sym.Variable('data')
    score = mx.sym.Variable('score')
    score_exp = mx.sym.expand_dims(score,axis=2)
    label = mx.sym.Variable('softmax_label')
    embed = mx.sym.Embedding(data=data, input_dim=input_size, weight=embed_weight, output_dim=num_embed, name='embed')
    if fix_embed:
        embed = mx.sym.BlockGrad(embed)
    Concat_input = mx.sym.Concat(embed,score_exp,dim=2,name='Concat')

    if enable_mask:
        data_mask_len = mx.sym.Variable('data_mask_len')
        Concat_input = mx.sym.SwapAxis(Concat_input, dim1=0, dim2=1)
        Concat_input = mx.sym.SequenceMask(data=Concat_input, use_sequence_length=True, sequence_length=data_mask_len, value=0.)
        Concat_input = mx.sym.SwapAxis(Concat_input, dim1=0, dim2=1)

    lstm_input = mx.sym.SliceChannel(data=Concat_input, num_outputs=seq_len, squeeze_axis=1)

    # forward
    forward_hidden = []
    for seqidx in range(seq_len):
        hidden = lstm_input[seqidx]
        hidden = mx.sym.FullyConnected(data=hidden,num_hidden = num_hidden,weight=fc_weight,bias=fc_bias,name='FC_%d'%seqidx)
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)

        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_lstm_o, indata=hidden,
                              prev_state=forward_last_states[i],
                              param=forward_param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden = next_state.h
            forward_last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        forward_hidden.append(hidden)

    # forward
    backward_hidden = []
    for seqidx in range(seq_len):
        k = seq_len - seqidx - 1
        hidden = lstm_input[k]
        hidden = mx.sym.FullyConnected(data=hidden,num_hidden = num_hidden,weight=fc_weight,bias=fc_bias,name='FC_%d'%k)
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)

        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_lstm_o, indata=hidden,
                              prev_state=backward_last_states[i],
                              param=forward_param_cells[i],
                              seqidx=k, layeridx=i, dropout=dp_ratio)
            hidden = next_state.h
            backward_last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        backward_hidden.insert(0, hidden)

    if enable_mask:
        forward_hidden_Concat = mx.sym.Concat(*forward_hidden, dim=0)
        backward_hidden_Concat = mx.sym.Concat(*backward_hidden, dim=0)
        hidden_Concat = mx.sym.broadcast_add(forward_hidden_Concat, backward_hidden_Concat)
        pred_all = mx.sym.FullyConnected(data=hidden_Concat, num_hidden=num_label, weight=cls_weight, bias=cls_bias, name='pred')
        pred_all = mx.sym.Reshape(data=pred_all, shape=(seq_len, -1, num_label))
        pred = mx.sym.SequenceLast(data=pred_all, sequence_length=data_mask_len, use_sequence_length=True)
    else:
        hidden_final = forward_hidden[-1] + backward_hidden[-1]
        pred = mx.sym.FullyConnected(data=hidden_final, num_hidden=num_label, weight=cls_weight, bias=cls_bias, name='pred')

    if output_type == 'sigmoid':
        sm = mx.sym.LogisticRegressionOutput(data=pred, label=label, name='softmax')
    else:
        label = mx.sym.Reshape(data=label, shape=(-1,))
        sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    if enable_mask:
        provide_data = ('data', 'score', 'data_mask_len', )
    else:
        provide_data = ('data', 'score', )
    init_states =  ['forward_l%d_init_c'%l  for l in range(num_lstm_layer)]
    init_states += ['forward_l%d_init_h'%l  for l in range(num_lstm_layer)]
    init_states += ['backward_l%d_init_c'%l for l in range(num_lstm_layer)]
    init_states += ['backward_l%d_init_h'%l for l in range(num_lstm_layer)]

    return sm, provide_data+tuple(init_states), ('softmax_label',)


def lstmn_unroll(num_lstm_layer, seq_len, input_size,
                num_hidden, num_embed,num_lstm_o, num_label, output_type='softmax', dropout=0., fix_embed=False, enable_mask=False):

    att_h_weight = mx.sym.Variable("att_h_weight", shape=(num_hidden, num_hidden))
    att_x_weight = mx.sym.Variable("att_x_weight", shape=(num_hidden, num_hidden))
    att_hatt_weight = mx.sym.Variable("att_hatt_weight", shape=(num_hidden, num_hidden))
    att_v_weight = mx.sym.Variable("att_v_weight", shape=(num_hidden, 1))

    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    fc_weight = mx.sym.Variable("fc_weight")
    fc_bias= mx.sym.Variable("fc_bias")

    i = 0
    param_cell = LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                    i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                    h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                    h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i))
    last_state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                            h=mx.sym.Variable("l%d_init_h" % i))

    # embeding layer
    data = mx.sym.Variable('data')
    score = mx.sym.Variable('score')
    score_exp = mx.sym.expand_dims(score,axis=2)
    label = mx.sym.Variable('softmax_label')
    embed = mx.sym.Embedding(data=data, input_dim=input_size, weight=embed_weight, output_dim=num_embed, name='embed')
    if fix_embed:
        embed = mx.sym.BlockGrad(embed)
    Concat_input = mx.sym.Concat(embed,score_exp,dim=2,name='Concat')

    if enable_mask:
        data_mask_len = mx.sym.Variable('data_mask_len')
        Concat_input = mx.sym.SwapAxis(Concat_input, dim1=0, dim2=1)
        Concat_input = mx.sym.SequenceMask(data=Concat_input, use_sequence_length=True, sequence_length=data_mask_len, value=0.)
        Concat_input = mx.sym.SwapAxis(Concat_input, dim1=0, dim2=1)

    lstm_input = mx.sym.SliceChannel(data=Concat_input, num_outputs=seq_len, squeeze_axis=1)

    hidden_h_all = []
    hidden_c_all = []
    last_h_att = last_state.h
    for seqidx in range(seq_len):
        datax = lstm_input[seqidx]
        datax = mx.sym.FullyConnected(data=datax,num_hidden = num_hidden,weight=fc_weight,bias=fc_bias,name='FC_%d'%seqidx)
        if dropout > 0.:
            datax = mx.sym.Dropout(data=datax, p=dropout)

        # compute attention
        if seqidx > 0:
            att_all = []
            att_wx = mx.sym.dot(datax, att_x_weight) # (batch_size, hidden_size)
            att_whatt = mx.sym.dot(last_h_att, att_hatt_weight) # (batch_size, hidden_size)
            for attidx in range(seqidx):
                att_wh = mx.sym.dot(hidden_h_all[attidx], att_h_weight) # (batch_size, hidden_size)
                att_tanh = mx.sym.Activation(att_wh + att_wx + att_whatt, act_type="tanh") # (batch_size, hidden_size)
                att = mx.sym.dot(att_tanh, att_v_weight) # (batch_size, 1)
                att_all.append(att) # (seqidx, batch_size, 1)
            all_att = mx.sym.Concat(*att_all, dim=1) # (batch_size, seqidx)
            s_att = mx.sym.SoftmaxActivation(all_att) # (batch_size, seqidx)
            s_att = mx.sym.Reshape(data=s_att, shape=(-1, seqidx, 1)) # (batch_size, seqidx, 1)

            Concat_h = mx.sym.Concat(*hidden_h_all, dim=1) # (batch_size, seqidx*hidden_size)
            Concat_h = mx.sym.Reshape(data=Concat_h, shape=(-1, seqidx, num_hidden)) # (batch_size, seqidx, hidden_size)
            h_att = mx.sym.broadcast_mul(s_att, Concat_h) # (batch_size, seqidx, hidden_size)
            h_att = mx.sym.sum(h_att, axis=1) # (batch_size, hidden_size)
            last_h_att = h_att
            Concat_c = mx.sym.Concat(*hidden_c_all, dim=1) # (batch_size, seqidx*hidden_size)
            Concat_c = mx.sym.Reshape(data=Concat_c, shape=(-1, seqidx, num_hidden)) # (batch_size, seqidx, hidden_size)
            c_att = mx.sym.broadcast_mul(s_att, Concat_c) # (batch_size, seqidx, hidden_size)
            c_att = mx.sym.sum(c_att, axis=1) # (batch_size, hidden_size)
            last_state = LSTMState(c=c_att, h=h_att)

        dp_ratio = dropout
        next_state = lstm(num_lstm_o, indata=datax,
                            prev_state=last_state,
                            param=param_cell,
                            seqidx=seqidx, layeridx=i, dropout=dp_ratio)
        hidden_h = next_state.h
        hidden_c = next_state.c

        # decoder
        if dropout > 0.:
            hidden_h = mx.sym.Dropout(data=hidden_h, p=dropout)
        hidden_h_all.append(hidden_h)
        hidden_c_all.append(hidden_c)

    if enable_mask:
        hidden_Concat = mx.sym.Concat(*hidden_h_all, dim=0)
        pred_all = mx.sym.FullyConnected(data=hidden_Concat, num_hidden=num_label, weight=cls_weight, bias=cls_bias, name='pred')
        pred_all = mx.sym.Reshape(data=pred_all, shape=(seq_len, -1, num_label))
        pred = mx.sym.SequenceLast(data=pred_all, sequence_length=data_mask_len, use_sequence_length=True)
    else:
        hidden_final = hidden_h_all[-1]
        pred = mx.sym.FullyConnected(data=hidden_final, num_hidden=num_label, weight=cls_weight, bias=cls_bias, name='pred')

    if output_type == 'sigmoid':
        sm = mx.sym.LogisticRegressionOutput(data=pred, label=label, name='softmax')
    else:
        label = mx.sym.Reshape(data=label, shape=(-1,))
        sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    if enable_mask:
        provide_data = ('data', 'score', 'data_mask_len', )
    else:
        provide_data = ('data', 'score', )
    init_states = ['l%d_init_c'%l for l in range(num_lstm_layer)]
    init_states += ['l%d_init_h'%l for l in range(num_lstm_layer)]

    return sm, provide_data+tuple(init_states), ('softmax_label',)


# ##################################
CNNParam = namedtuple("CNNParam", ["conv_weight", "conv_bias"])
CNNModel = namedtuple("CNNModel", ['cnn_exec', 'symbol', 'data', 'label', 'param_blocks'])

def text_cnn(seq_len, num_embed, input_size,
        num_label, filter_list, num_filter, output_type='softmax', dropout=0., fix_embed=False):

    input_x = mx.sym.Variable('data') #(batch_size, seq_len)
    label = mx.sym.Variable('softmax_label')
    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')
    embed_weight = mx.sym.Variable("embed_weight")

    param_cells = []
    for i, filter_size in enumerate(filter_list):
        param_cells.append(CNNParam(conv_weight=mx.sym.Variable("conv%d_weight" % i),
                                    conv_bias=mx.sym.Variable("conv%d_bias" % i)))

    # embedding layer
    embed = mx.sym.Embedding(data=input_x, input_dim=input_size, weight=embed_weight,
                             output_dim=num_embed, name='embed') #(batch_size, seq_len, num_embed)
    if fix_embed:
        embed = mx.sym.BlockGrad(embed)
    conv_input = mx.sym.Reshape(data=embed, shape=(-1, 1, seq_len, num_embed)) #(batch_size, 1, seq_len, num_embed)

    # create convolution + (max) pooling layer for each filter operation
    pooled_outputs = []
    for i, filter_size in enumerate(filter_list):
        convi = mx.sym.Convolution(data=conv_input,
                                   weight=param_cells[i].conv_weight,
                                   bias=param_cells[i].conv_bias,
                                   kernel=(filter_size, num_embed),
                                   num_filter=num_filter) # (batch_size, num_filter, seq_len-filter_size+1, 1)
        relui = mx.sym.Activation(data=convi, act_type='relu')
        pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(seq_len - filter_size + 1, 1), stride=(1,1))
        pooled_outputs.append(pooli)
    # combine all pooled outputs
    total_filters = num_filter * len(filter_list)
    Concat = mx.sym.Concat(*pooled_outputs, dim=1)
    h_pool = mx.sym.Reshape(data=Concat, shape=(-1, total_filters))
    # dropout layer
    if dropout > 0.0:
        h_pool = mx.sym.Dropout(data=h_pool, p=dropout)
    # fully connected
    pred = mx.sym.FullyConnected(data=h_pool, weight=cls_weight, bias=cls_bias, num_hidden=num_label)

    if output_type != 'sigmoid':
        label = mx.sym.Reshape(data=label, shape=(-1,))

    # output
    if output_type == 'sigmoid':
        sm = mx.sym.LogisticRegressionOutput(data=pred, label=label, name='softmax')
    else:
        #sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax', normalization='batch')
        sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    return sm, ('data',), ('softmax_label',)


# ##################################
def bowlstm_unroll(num_lstm_layer, seq_len, input_size,
                num_hidden, num_embed,num_lstm_o, num_label, output_type='softmax', dropout=0.):

    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    fc_weight = mx.sym.Variable("fc_weight")
    fc_bias= mx.sym.Variable("fc_bias")
    fc1_weight = mx.sym.Variable("fc1_weight")
    fc1_bias= mx.sym.Variable("fc1_bias")
    fc2_weight = mx.sym.Variable("fc2_weight")
    fc2_bias= mx.sym.Variable("fc2_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    data = mx.sym.Variable('data') # (batch_size, seq_len, input_size)
    data = mx.sym.Reshape(data, shape=(-1, input_size))
    data_fc = mx.sym.FullyConnected(data=data, num_hidden=num_hidden, weight=fc1_weight, bias=fc1_bias)
    data_fc = mx.sym.Reshape(data_fc, shape=(-1, seq_len, num_hidden))

    data_act = mx.sym.Variable('data_act') # (batch_size, seq_len, input_size)
    data_act = mx.sym.Reshape(data_act, shape=(-1, input_size))
    data_act_fc = mx.sym.FullyConnected(data=data_act, num_hidden=num_hidden, weight=fc2_weight, bias=fc2_bias)
    data_act_fc = mx.sym.Reshape(data_act_fc, shape=(-1, seq_len, num_hidden))

    Concat_input = mx.sym.Concat(data_fc, data_act_fc, dim=2, name='Concat') # (batch_size, seq_len, num_hidden*2)
    if dropout > 0.:
        Concat_input = mx.sym.Dropout(data=Concat_input, p=dropout)
    lstm_input = mx.sym.SliceChannel(data=Concat_input, num_outputs=seq_len, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len):
        hidden = lstm_input[seqidx]
        hidden = mx.sym.FullyConnected(data=hidden, num_hidden=num_hidden, weight=fc_weight, bias=fc_bias) # (batch_size, num_hidden)
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_lstm_o, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden = next_state.h # (batch_size, num_hidden)
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    hidden_Concat = mx.sym.Concat(*hidden_all, dim=0) # (seq_len*batch_size, num_hidden)
    pred = mx.sym.FullyConnected(data=hidden_Concat, num_hidden=num_label, weight=cls_weight, bias=cls_bias, name='pred')
    pred = mx.sym.Reshape(data=pred, shape=(seq_len, -1, num_label))
    pred = mx.sym.SwapAxis(data=pred, dim1=0, dim2=1) # (batch_size, seq_len, num_label)

    label = mx.sym.Variable('softmax_label') # (batch_size, seq_len, label_out)
    if output_type == 'sigmoid':
        sm = mx.sym.LogisticRegressionOutput(data=pred, label=label, name='softmax')
    else:
        label = mx.sym.Reshape(data=label, shape=(-1, seq_len))
        sm = mx.sym.SoftmaxOutput(data=pred, label=label, preserve_shape=True, name='softmax')

    init_states = ['l%d_init_c'%l for l in range(num_lstm_layer)]
    init_states += ['l%d_init_h'%l for l in range(num_lstm_layer)]

    return sm, ('data', 'data_act',)+tuple(init_states), ('softmax_label',)


#cnnlstm
def cnnlstm_unroll(num_lstm_layer, seq_len, input_size,
                    num_hidden, num_embed, num_lstm_o, num_label, filter_list, num_filter,
                    max_nbest, max_sentlen, output_type='softmax', dropout=0.):

     embed_weight = mx.sym.Variable("embed_weight")
     cls_weight = mx.sym.Variable("cls_weight")
     cls_bias = mx.sym.Variable("cls_bias")
     fc_weight = mx.sym.Variable("fc_weight")
     fc_bias= mx.sym.Variable("fc_bias")
     fc1_weight = mx.sym.Variable("fc1_weight")
     fc1_bias= mx.sym.Variable("fc1_bias")
     fc2_weight = mx.sym.Variable("fc2_weight")
     fc2_bias= mx.sym.Variable("fc2_bias")
     param_cells = []
     last_states = []
     for i in range(num_lstm_layer):
         param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                      i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                      h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                      h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
         state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                           h=mx.sym.Variable("l%d_init_h" % i))
         last_states.append(state)
     assert(len(last_states) == num_lstm_layer)

     # Step1: construct data cnn model
     cnnparam_cells = []
     for i, filter_size in enumerate(filter_list):
         cnnparam_cells.append(CNNParam(conv_weight=mx.sym.Variable("conv%d_weight" % i),
                                     conv_bias=mx.sym.Variable("conv%d_bias" % i)))
     data = mx.sym.Variable('data') # (batch_size, seq_len, max_nbest, max_sentlen)
     score = mx.sym.Variable('score') # (batch_size, seq_len, max_nbest)
     # embedding layer
     embed = mx.sym.Embedding(data=data, input_dim=input_size, weight=embed_weight,
                              output_dim=num_embed, name='embed') # (batch_size, seq_len, max_nbest, max_sentlen, num_embed)
     conv_input = mx.sym.Reshape(data=embed, shape=(-1, 1, max_sentlen, num_embed)) # (batch_size*seq_len*max_nbest, 1, max_sentlen, num_embed)
     # create convolution + (max) pooling layer for each filter operation
     pooled_outputs = []
     for i, filter_size in enumerate(filter_list):
         convi = mx.sym.Convolution(data=conv_input,
                                    weight=cnnparam_cells[i].conv_weight,
                                    bias=cnnparam_cells[i].conv_bias,
                                    kernel=(filter_size, num_embed),
                                    num_filter=num_filter) # (batch_size*seq_len*max_nbest, num_filter, max_sentlen-filter_size+1, 1)
         relui = mx.sym.Activation(data=convi, act_type='relu')
         pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(max_sentlen - filter_size + 1, 1), stride=(1,1))
         pooled_outputs.append(pooli)
     # combine all pooled outputs
     total_filters = num_filter * len(filter_list)
     Concat = mx.sym.Concat(*pooled_outputs, dim=1)
     h_pool = mx.sym.Reshape(data=Concat, shape=(-1, total_filters)) # (batch_size*seq_len*max_nbest, total_filters)
     # dropout layer
     if dropout > 0.0:
         h_pool = mx.sym.Dropout(data=h_pool, p=dropout)
     score_exp = mx.sym.Reshape(data=score, shape=(-1, 1)) # (batch_size*seq_len*max_nbest, 1)

     # TODO how to deal with nbest information properly ?
     #Concat_data = mx.sym.Concat(h_pool, score_exp, dim=1) # (batch_size*seq_len*max_nbest, total_filters+1)
     Concat_data = mx.sym.broadcast_mul(h_pool, score_exp)
     data_fc = mx.sym.FullyConnected(data=Concat_data, num_hidden=num_hidden, weight=fc1_weight, bias=fc1_bias)
     data_fc = mx.sym.Reshape(data=data_fc, shape=(-1, seq_len, max_nbest, num_hidden))
     data_fc = mx.sym.sum(data=data_fc, axis=2) # (batch_size, seq_len, num_hidden)

     # Step2: construct data_act fc model
     data_act = mx.sym.Variable('data_act') # (batch_size, seq_len, input_size)
     data_act = mx.sym.Reshape(data_act, shape=(-1, input_size))
     data_act_fc = mx.sym.FullyConnected(data=data_act, num_hidden=num_hidden, weight=fc2_weight, bias=fc2_bias)
     data_act_fc = mx.sym.Reshape(data_act_fc, shape=(-1, seq_len, num_hidden))

     # Step3: construct lstm model
     Concat_input = mx.sym.Concat(data_fc, data_act_fc, dim=2, name='Concat') # (batch_size, seq_len, num_hidden*2)
     if dropout > 0.:
         Concat_input = mx.sym.Dropout(data=Concat_input, p=dropout)
     lstm_input = mx.sym.SliceChannel(data=Concat_input, num_outputs=seq_len, squeeze_axis=1)

     hidden_all = []
     for seqidx in range(seq_len):
         hidden = lstm_input[seqidx]
         hidden = mx.sym.FullyConnected(data=hidden, num_hidden=num_hidden, weight=fc_weight, bias=fc_bias)
         if dropout > 0.:
             hidden = mx.sym.Dropout(data=hidden, p=dropout)
         # stack LSTM
         for i in range(num_lstm_layer):
             if i == 0:
                 dp_ratio = 0.
             else:
                 dp_ratio = dropout
             next_state = lstm(num_lstm_o, indata=hidden,
                               prev_state=last_states[i],
                               param=param_cells[i],
                               seqidx=seqidx, layeridx=i, dropout=dp_ratio)
             hidden = next_state.h
             last_states[i] = next_state
         # decoder
         if dropout > 0.:
             hidden = mx.sym.Dropout(data=hidden, p=dropout)
         hidden_all.append(hidden)

     hidden_Concat = mx.sym.Concat(*hidden_all, dim=0) # (seq_len*batch_size, num_hidden)
     pred = mx.sym.FullyConnected(data=hidden_Concat, num_hidden=num_label, weight=cls_weight, bias=cls_bias, name='pred')
     pred = mx.sym.Reshape(data=pred, shape=(seq_len, -1, num_label))
     pred = mx.sym.SwapAxis(data=pred, dim1=0, dim2=1) # (batch_size, seq_len, num_label)

     label = mx.sym.Variable('softmax_label') # (batch_size, seq_len, label_out)
     if output_type == 'sigmoid':
         sm = mx.sym.LogisticRegressionOutput(data=pred, label=label, name='softmax')
     else:
         label = mx.sym.Reshape(data=label, shape=(-1, seq_len))
         sm = mx.sym.SoftmaxOutput(data=pred, label=label, preserve_shape=True, name='softmax')

     init_states = ['l%d_init_c'%l for l in range(num_lstm_layer)]
     init_states += ['l%d_init_h'%l for l in range(num_lstm_layer)]

     return sm, ('data', 'data_act', 'score')+tuple(init_states), ('softmax_label',)

class LayerNormalization:
    """
    Implements Ba et al, Layer Normalization (https://arxiv.org/abs/1607.06450).
    :param num_hidden: Number of hidden units of layer to be normalized.
    :param prefix: Optional prefix of layer name.
    :param scale: Optional variable for scaling of shape (num_hidden,). Will be created if None.
    :param shift: Optional variable for shifting of shape (num_hidden,). Will be created if None.
    :param scale_init: Initial value of scale variable if scale is None. Default 1.0.
    :param shift_init: Initial value of shift variable if shift is None. Default 0.0.
    """

    # TODO(fhieber): this should eventually go to MXNet

    def __init__(self,
                 num_hidden,
                 prefix = None,
                 scale= None,
                 shift = None,
                 scale_init= 1.0,
                 shift_init= 0.0):
        #utils.check_condition(num_hidden > 1,
        #                      "Layer normalization should only be applied to layers with more than 1 neuron.")
        self.prefix = prefix
        self.scale = scale if scale is not None else mx.sym.Variable('%s_gamma' % prefix, shape=(num_hidden,),
                                                                     init=mx.init.Constant(value=scale_init))
        self.shift = shift if shift is not None else mx.sym.Variable('%s_beta' % prefix, shape=(num_hidden,),
                                                                     init=mx.init.Constant(value=shift_init))

    @staticmethod
    def moments(inputs):
        """
        Computes mean and variance of the last dimension of a Symbol.
        :param inputs: Shape: (d0, ..., dn, hidden).
        :return: mean, var: Shape: (d0, ..., dn, 1).
        """
        mean = mx.sym.mean(data=inputs, axis=-1, keepdims=True)
        # TODO(fhieber): MXNet should have this.
        var = mx.sym.mean(mx.sym.square(mx.sym.broadcast_minus(inputs, mean)), axis=-1, keepdims=True)
        return mean, var

    def normalize(self, inputs, eps = 0.000001):
        """
        Normalizes hidden units of inputs as follows:
        inputs = scale * (inputs - mean) / sqrt(var + eps) + shift
        Normalization is performed over the last dimension of the input data.
        :param inputs: Inputs to normalize. Shape: (d0, ..., dn, num_hidden).
        :param eps: Variance epsilon.
        :return: inputs_norm: Normalized inputs. Shape: (d0, ..., dn, num_hidden).
        """
        #eps=mx.sym.full((32,seq_len,num_label),1e-14)
        mean, var = self.moments(inputs)
        inputs_norm = mx.sym.broadcast_minus(inputs, mean, name='%sinp_minus_mean' % self.prefix)
        inputs_norm = mx.sym.broadcast_mul(inputs_norm, mx.sym.rsqrt(var + eps), name='%sinp_norm' % self.prefix)
        inputs_norm = mx.sym.broadcast_mul(inputs_norm, self.scale, name='%sinp_norm_scaled' % self.prefix)
        inputs_norm = mx.sym.broadcast_add(inputs_norm, self.shift, name='%sinp_norm_scaled_shifted' % self.prefix)
        return inputs_norm


ResParam = namedtuple("ResParam", ["conv1_weight", "conv1_bias",
                                     "conv3_weight", "conv3_bias","g3","b3","mm3","mv3","b2","g2","mm2","mv2"])
BNLSTMParam = namedtuple("BNLSTMParam", ["i2h_weight", "i2h_bias",
                                        "h2h_weight", "h2h_bias","g5","mm5","mv5","g6","mm6","mv6","g7","b7","mm7","mv7"])


CParam = namedtuple("CParam", ["cfc_weight","conv_weight","conv_bias"])#,"g","b","mm","mv"])

C2Param = namedtuple("C2Param", ["cfc_weight","conv_weight","conv_bias","g","b","mm","mv"])


#double
def doublelstm_unroll(num_lstm_layer, seq_len, input_size,
                   num_hidden, num_embed, num_lstm_o, num_label, filter_list, num_filter,
                   max_nbest, max_sentlen, output_type='softmax', dropout=0.):

    #embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    fc_weight = mx.sym.Variable("fc_weight")
    fc_bias= mx.sym.Variable("fc_bias")
    fc1_weight = mx.sym.Variable("fc1_weight")
    fc1_bias= mx.sym.Variable("fc1_bias")
    fc2_weight = mx.sym.Variable("fc2_weight")
    fc2_bias= mx.sym.Variable("fc2_bias")
    fc3_weight = mx.sym.Variable("fc3_weight")
    fc3_bias= mx.sym.Variable("fc3_bias")
    

    conv_weight = mx.sym.Variable("conv_weight")
    conv_bias = mx.sym.Variable("conv_bias")
    conv2_weight = mx.sym.Variable("conv2_weight")
    conv2_bias = mx.sym.Variable("conv2_bias")
   

    forward_param_cells = []
    backward_param_cells = []
    forward_last_states = []
    backward_last_states = []
    for i in range(num_lstm_layer):
         # backward
        prefix = "backward_"
        backward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable(prefix+"l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable(prefix+"l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable(prefix+"l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable(prefix+"l%d_h2h_bias" % i)))
        backward_state = LSTMState(c=mx.sym.Variable(prefix+"l%d_init_c" % i),
                          h=mx.sym.Variable(prefix+"l%d_init_h" % i))
        backward_last_states.append(backward_state)
    assert(len(backward_last_states) == num_lstm_layer)
    
    
    num_hidden1 = num_hidden
    data_act = mx.sym.Variable('data_act') 
    data_act = mx.sym.SliceChannel(data=data_act,axis=1, num_outputs=seq_len, squeeze_axis=1)
    data = mx.sym.Variable('data') # (batch_size, seq_len,2, max_nbest, max_sentlen)
    
    g1=mx.sym.Variable("bn1_gamma")
    b1=mx.sym.Variable("bn1_beta")
    mv1=mx.sym.Variable("bn1_moving_var")
    mm1=mx.sym.Variable("bn1_moving_mean")

    numC=4
    paramC=[]
     
    for i in range(numC):
        paramC.append(CParam(cfc_weight=mx.sym.Variable("cfc%d_weight"%i),
                            conv_weight = mx.sym.Variable("cconv%d_weight"%i),
                            conv_bias = mx.sym.Variable("cconv%d_bias"%i),
                             ))

    paramC2=[]
    numC2=4 
    for i in range(numC2):
        paramC2.append(C2Param(cfc_weight=mx.sym.Variable("cfg%d_weight"%i),
                            conv_weight = mx.sym.Variable("gconv%d_weight"%i),
                            g=mx.sym.Variable("bn5%d_gamma"%i),
                            b=mx.sym.Variable("bn5%d_beta"%i),
                            mm=mx.sym.Variable("bn5%d_moving_mean"%i),
                            mv=mx.sym.Variable("bn5%d_moving_var"%i),
                            conv_bias = mx.sym.Variable("gconv%d_bias"%i),
                             ))
    ln=LayerNormalization(128*4,"ln1")

    slot = mx.sym.Variable('slot') # (batch_size,nnl,300)
    
    slota= mx.sym.SliceChannel(data=slot,axis=1, num_outputs=len(num_label), squeeze_axis=1)

    valuea=[]
    vt=()
    for i in range(len(num_label)):
        valuea.append(mx.sym.Variable('value_%d'%i)) # (batch_size,num_label,300)
        vt+=('value_%d'%i,) 
     
    label = mx.sym.Variable('softmax_label') # (batch_size, seq_len, label_out)
    
    labela= mx.sym.SliceChannel(data=label,axis=2, num_outputs=len(num_label), squeeze_axis=1)


    #scored_data= mx.sym.SliceChannel(data=data,axis=2, num_outputs=2, squeeze_axis=1)
    #f=scored_data[0]
    #fs=scored_data[1]
    lstm_input1= mx.sym.SliceChannel(data=data, axis=1,num_outputs=seq_len, squeeze_axis=1)
######

    # embed= mx.sym.SliceChannel(data=embed,axis=2, num_outputs=2, squeeze_axis=1)
    # Concat_input= mx.sym.SliceChannel(data=embed[0],axis=2, num_outputs=max_nbest, squeeze_axis=1)

    # lstm_input1= mx.sym.SliceChannel(data=Concat_input[0], axis=1,num_outputs=seq_len, squeeze_axis=1)
    # hidden_all = []
    nf=128
    #hidden_all = []
    ksize=3
    npad=(ksize-1)/2
    

    bm=backward_last_states[0]
    cea=[]
    sout=[]
    for k in range(len(num_label)):
        slot=slota[k] 
        value=valuea[k]
        label=labela[k]
        val_embed= mx.sym.SliceChannel(data=value,axis=0, num_outputs=32, squeeze_axis=1)
        val_embed=val_embed[0]
    
        hidden_all = []
        for seqidx in range(seq_len):
            
            input1 = lstm_input1[seqidx]
            lstm_input2= mx.sym.SliceChannel(data=input1, axis=1,num_outputs=max_sentlen, squeeze_axis=1)
            hidden_all2 =[]
            hidden_all3=[]
            for i in range(numC):
                
                xout=[]   
                xout2=[] 
                for seqidx2 in range(max_sentlen):
                    x = lstm_input2[seqidx2]
                    tmp=mx.sym.FullyConnected(data=x,num_hidden = 128,weight=paramC[i].cfc_weight,no_bias=0,bias=paramC[i].conv_bias) 
                    tmp=mx.sym.expand_dims(tmp,axis=1)
                    xout.append(tmp)
                    if seqidx2>0:
                        tmp1= mx.sym.Concat(mx0,x, dim=1)
                        tmp1=mx.sym.FullyConnected(data=tmp1,num_hidden = 128,weight=paramC2[i].cfc_weight,no_bias=0,bias=paramC2[i].conv_bias) 
                        tmp1=mx.sym.expand_dims(tmp1,axis=1)
                        xout2.append(tmp1)
                    mx0=x
                x= mx.sym.Concat(*xout, dim=1)
                
                xm= mx.sym.Concat(*xout2, dim=1)

                x =mx.sym.sum(x,axis=1,keepdims=1)

                xm =mx.sym.sum(xm,axis=1,keepdims=1)

                hidden_all2.append(x)
                hidden_all3.append(xm)
            x= mx.sym.Concat(*hidden_all2, dim=2)
            xg= mx.sym.Concat(*hidden_all3, dim=2)
            x=x+xg
           
            x=ln.normalize(x)
         

            x= mx.sym.Activation(data=x, act_type="relu")
           
            data_act_fc = mx.sym.FullyConnected(data=data_act[seqidx], num_hidden=num_hidden, weight=fc2_weight, bias=fc2_bias)
            data_act_fc =mx.sym.expand_dims(data_act_fc,axis=1)
            data_act_fc= mx.sym.Activation(data=data_act_fc, act_type="relu")
  
 
            x = mx.sym.FullyConnected(data=x, num_hidden=128, weight=fc1_weight, bias=fc1_bias)
            x =mx.sym.expand_dims(x,axis=1)
            
            x = mx.sym.Concat(x, data_act_fc, dim=2, name='Concat') # (batch_size, num_hidden*2) 
            
            slotx = mx.sym.FullyConnected(data=slot, num_hidden=256, weight=fc3_weight, bias=fc3_bias)
            slotx= mx.sym.Activation(data=slotx, act_type="relu")
            slotx =mx.sym.expand_dims(slotx,axis=1)
            x1=mx.sym.broadcast_mul(x,slotx)
           


            if dropout > 0.:
                x1 = mx.sym.Dropout(data=x1, p=dropout)

            for j in range(num_lstm_layer):
                if j == 0:
                    dp_ratio = 0.
                else:
                    dp_ratio = dropout
                next_state = lstm(num_hidden1, indata=x1,
                                  prev_state=backward_last_states[j],
                                  param=backward_param_cells[j],
                                  seqidx=seqidx, layeridx=j, dropout=dp_ratio)#,ln3=ln3,ln4=ln4,ln5=ln5)
                hidden = next_state.h
                backward_last_states[j] = next_state
            if dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=dropout)
            
            #hidden=mx.sym.Concat(hidden,hidden_final,dim=1)
	        
            #hidden=hidden+hidden_final
	        
            #hidden= mx.sym.Reshape(hidden, shape=(seq_len, 382, num_hidden))
            hidden_all.append(hidden)
        
        backward_last_states[0]=bm
        hidden_Concat = mx.sym.Concat(*hidden_all, dim=0) # (seq_len*batch_size, num_hidden)
        pred = mx.sym.FullyConnected(data=hidden_Concat, num_hidden=300, weight=cls_weight, bias=cls_bias, name='pred')
        
        pred= mx.sym.Activation(data=pred, act_type="relu")
        pred = mx.sym.Reshape(data=pred, shape=(seq_len, 32, 300),name='sd')
        
        pred= mx.sym.SliceChannel(data=pred, axis=0,num_outputs=seq_len, squeeze_axis=1)
        value = mx.sym.SwapAxis(data=value, dim1=1, dim2=2) # (batch_size, 300, n_la)
        #value =mx.sym.Reshape(data=value,shape=(-1,num_label))
        outp=[]
        for p in pred: 
            p = mx.sym.expand_dims(data=p, axis=2)
            
            #eucl
            tmp=mx.sym.square(mx.sym.broadcast_sub(value,p))  
            tmp=-mx.sym.sqrt(mx.sym.sum(tmp,axis=1,keepdims=1))
            
            #manha
            #tmp=mx.sym.abs(mx.sym.broadcast_sub(value,p))  
            #tmp=-mx.sym.sum(tmp,axis=1,keepdims=1)
 
            outp.append(tmp)
        
        pred= mx.sym.Concat(*outp, dim=1) # (batch_size,seq_len, num_la)

  
        label = mx.sym.Reshape(data=label, shape=(-1,seq_len))
        
        softmax = mx.sym.softmax(data=pred)
        
        sf = mx.sym.BlockGrad(softmax)
        sf =mx.sym.argmax(sf,axis=2,keepdims=1) 
        softmax_output = mx.sym.BlockGrad(sf)
        ce = mx.sym.SoftmaxOutput(data=pred, label=label, preserve_shape=True)
        cea.append(ce)
        sout.append(softmax_output) 
   
    ce = mx.sym.Group(cea)
    sout = mx.sym.Concat(*sout, dim=2,name='softmax') 
    sm = mx.sym.Group([sout,ce])
 
    init_states = ['backward_l%d_init_c'%l for l in range(num_lstm_layer)]
    init_states += ['backward_l%d_init_h'%l for l in range(num_lstm_layer)]
    return sm, ('data', 'data_act','slot')+vt+tuple(init_states), ('softmax_label',)


def MatRNN(num_hidden, f, fs, prev_state, param, seqidx, layeridx, dropout,num_label,output_type,label,seq_len):
    #Mat's RNN Model without engineered features of fs, fv.
    #if dropout > 0.:
    #    indata = mx.sym.Dropout(data=indata, p=dropout)
    
    Concat_state = mx.sym.Concat(prev_state.c,prev_state.h,dim=1)
    
    Concat_state = mx.sym.Concat(Concat_state,f,dim=1)
    
    Concat_m=mx.sym.Concat(prev_state.c,f,dim = 1)

    g=mx.sym.Concat(f,fs,dim=1)
    g=mx.sym.Concat(g,prev_state.h,dim=1)
    g=mx.sym.Concat(g,prev_state.c,dim=1)
    
    #Concat_state = mx.sym.Reshape(Concat_state, shape=(32, seq_len,2, num_hidden))
    i2h = mx.sym.FullyConnected(data=Concat_state,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_label,
                                name="mt%d_l%d_i2h" % (seqidx, layeridx))

    i2g = mx.sym.FullyConnected(data=g,
                                weight=param.i2g_weight,
                                bias=param.i2g_bias,
                                num_hidden=num_label,
                                name="mt%d_l%d_i2g" % (seqidx, layeridx))
    

    h2h = mx.sym.FullyConnected(data=Concat_m,
                                weight=param.h2h_weight,
                                no_bias=True,
                                num_hidden=num_hidden ,
                                name="mt%d_l%d_h2h" % (seqidx, layeridx))
    if dropout > 0.:
        h2h = mx.sym.Dropout(data=h2h, p=dropout)
    

    h = mx.sym.Activation(i2h, act_type="relu")
    g = mx.sym.Activation(i2g, act_type="relu")
    p=h+g

    #label = mx.sym.SwapAxis(data=label, dim1=0, dim2=1) 

    label = mx.sym.SliceChannel(data=label,axis=1, num_outputs=seq_len, squeeze_axis=1)
    if output_type == 'sigmoid':
        next_p = mx.sym.LogisticRegressionOutput(data=p, label=label[seqidx], name='softmax')
    else:
        label = mx.sym.Reshape(data=label[seqidx], shape=(-1,))
        next_p = mx.sym.SoftmaxOutput(data=p, label=label, preserve_shape=True, name='softmax')

    # next_p = mx.sym.SoftmaxOutput(p, act_type="softmax")
    #next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_m = mx.sym.Activation(h2h, act_type="relu")
    return LSTMState(c=next_m, h=next_p)


#reslstm
def reslstm_unroll(num_lstm_layer, seq_len, input_size,
                   num_hidden, num_embed, num_lstm_o, num_label, filter_list, num_filter,
                   max_nbest, max_sentlen, output_type='softmax', dropout=0.):
    lrm=1.0
    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight",lr_mult=lrm)
    cls_bias = mx.sym.Variable("cls_bias",lr_mult=lrm)
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i, lr_mult=lrm),
                                      i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i, lr_mult=lrm),
                                      h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i,lr_mult=lrm),
                                      h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i,lr_mult=lrm),))
                                    
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                           h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)
    #M0=mx.sym.Variable("m_init_m0")
    mems=[] 
    numM=2
    for i in range(numM):
        m=mx.sym.Variable("m_init_m%d"%i)
        mems.append(m)
        

    scored_data = mx.sym.Variable('data') # (batch_size, seq_len,2, input_size)
    data_act = mx.sym.Variable('data_act') # (batch_size, seq_len,2, input_size)
    
    #multn=mx.sym.Variable('multn')
    #sub=mx.sym.Variable('sub')


    scored_data= mx.sym.SliceChannel(scored_data,axis=2, num_outputs=2, squeeze_axis=0)
    data_act = mx.sym.SliceChannel(data_act,axis=2, num_outputs=2, squeeze_axis=0)
    Concat_input = mx.sym.Concat(scored_data[0], data_act[0], dim=2, name='Concat12') # (batch_size, seq_len, 2, input_size)

   
    #mask 
    #data_mask_len = mx.sym.Variable('data_mask_len')
   # Concat_input = mx.sym.SwapAxis(Concat_input, dim1=0, dim2=1)
   # Concat_input = mx.sym.SequenceMask(data=Concat_input, use_sequence_length=True, sequence_length=data_mask_len, value=0.)
   # Concat_input = mx.sym.SwapAxis(Concat_input, dim1=0, dim2=1)


    lstm_input = mx.sym.SliceChannel(data=Concat_input,axis=1, num_outputs=seq_len, squeeze_axis=1)
    
    label = mx.sym.Variable('softmax_label') # (batch_size, seq_len, label_out)
   # conv_weight = mx.sym.Variable("conv_weight")
   # conv_bias = mx.sym.Variable("conv_bias")
    ffc_weight = mx.sym.Variable("ffc_weight",lr_mult=lrm)
    ffc_bias = mx.sym.Variable("ffc_bias",lr_mult=lrm)
   
    numC=3
    #numL=1
    #paramL=[]
    #for j in range(numL):
       
    paramC=[]
     
    for i in range(numC):
        paramC.append(CParam(cfc_weight=mx.sym.Variable("cfc%d_weight"%i),
                            conv_weight = mx.sym.Variable("cconv%d_weight"%i),
                            g=mx.sym.Variable("bn4%d_gamma"%i),
                            b=mx.sym.Variable("bn4%d_beta"%i),
                            mm=mx.sym.Variable("bn4%d_moving_mean"%i),
                            mv=mx.sym.Variable("bn4%d_moving_var"%i),
                            conv_bias = mx.sym.Variable("cconv%d_bias"%i),
                             ))
    #paramL.append(paramC)
    paramC2=[]
     
    for i in range(numC):
        paramC2.append(C2Param(cfc_weight=mx.sym.Variable("cfg%d_weight"%i),
                            conv_weight = mx.sym.Variable("gconv%d_weight"%i),
                            g=mx.sym.Variable("bn5%d_gamma"%i),
                            b=mx.sym.Variable("bn5%d_beta"%i),
                            mm=mx.sym.Variable("bn5%d_moving_mean"%i),
                            mv=mx.sym.Variable("bn5%d_moving_var"%i),
                            conv_bias = mx.sym.Variable("gconv%d_bias"%i),
                             ))
    nf=16
    hidden_all = []
    ksize=15
    npad=(ksize-1)/2
    for seqidx in range(seq_len):
        
        x = lstm_input[seqidx]#(b,2,in)
        mm= mx.sym.Concat(mems[numM-1], mems[numM-2], dim=1)
        mems[numM-1]=mems[numM-2]
        for i in range(numM-2):
            mm=mx.sym.Concat(mm,mems[numM-3-i],dim=1)
            mems[numM-2-i]=mems[numM-3-i]
        mems[0]=x
        x1= mx.sym.Concat(mm, x, dim=1) # (batch_size, 8, in)
        

        #def caps(x,param,numC=8):
        x= mx.sym.SliceChannel(x1,axis=1, num_outputs=6, squeeze_axis=0)
        xout=[]    
        for i in range(numC):
            intx=[]    
            for j in range(6):
                tmp=mx.sym.FullyConnected(data=x[j],num_hidden = 256,weight=paramC[i].cfc_weight,no_bias=1) 
                tmp=mx.sym.expand_dims(tmp,axis=1)
                intx.append(tmp)
            outx= mx.sym.Concat(*intx, dim=1) 
            outx= mx.sym.Convolution(data=outx, kernel=(1,), stride=(1,),num_filter=1,
                                 weight=paramC[i].conv_weight,no_bias=0,bias=paramC[i].conv_bias)  #,bias=conv2_bias)
        
            outx=mx.sym.BatchNorm(data=outx,fix_gamma=False,momentum=0.9,gamma=paramC[i].g,beta=paramC[i].b,moving_var=paramC[i].mv,moving_mean=paramC[i].mm)
            #outx=squash(outx,-1)
            xout.append(outx)
        x= mx.sym.Concat(*xout, dim=1)
        x0= mx.sym.Activation(data=x, act_type="relu")
        x= mx.sym.SliceChannel(x0,axis=1, num_outputs=numC, squeeze_axis=0)
        xout=[]    
        for i in range(3):
            intx=[]    
            for j in range(3):
                tmp=mx.sym.FullyConnected(data=x[j],num_hidden = 256,weight=paramC2[i].cfc_weight,no_bias=1) 
                tmp=mx.sym.expand_dims(tmp,axis=1)
                intx.append(tmp)
            outx= mx.sym.Concat(*intx, dim=1) 
            outx= mx.sym.Convolution(data=outx, kernel=(1,), stride=(1,),num_filter=1,
                                 weight=paramC2[i].conv_weight,no_bias=0,bias=paramC2[i].conv_bias)  #,bias=conv2_bias)
        
            outx=mx.sym.BatchNorm(data=outx,fix_gamma=False,momentum=0.9,gamma=paramC2[i].g,beta=paramC2[i].b,moving_var=paramC2[i].mv,moving_mean=paramC2[i].mm)
            #outx=squash(outx,-1)
            xout.append(outx)
        x= mx.sym.Concat(*xout, dim=1)
        x=x+x0
        x= mx.sym.Activation(data=x, act_type="relu")
        x= mx.sym.FullyConnected(data=x,num_hidden = 256,weight=ffc_weight,bias=ffc_bias,name='FC_1', no_bias=0)
        #hidden= mx.sym.Activation(data=x, act_type="relu")
        for j in range(num_lstm_layer):
            if j == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_lstm_o, indata=x,
                              prev_state=last_states[j],
                              param=param_cells[j],
                              seqidx=seqidx, layeridx=j, dropout=dp_ratio)
            hidden = next_state.h
            last_states[j] = next_state
        #hidden= mx.sym.FullyConnected(data=hidden,num_hidden = num_hidden,weight=ffc1_weight,bias=ffc1_bias,name='FC')
        #if dropout > 0.:
        #    hidden = mx.sym.Dropout(data=hidden, p=dropout)
        
        hidden_all.append(hidden)
 
        
    hidden_Concat = mx.sym.Concat(*hidden_all, dim=0) # (seq_len*batch_size, num_hidden)
    pred = mx.sym.FullyConnected(data=hidden_Concat, num_hidden=num_label, weight=cls_weight, bias=cls_bias,name='pred')
    pred = mx.sym.Reshape(data=pred, shape=(seq_len, -1, num_label),name='sd')
    #mask
    #pred = mx.sym.SequenceLast(data=pred, sequence_length=data_mask_len, use_sequence_length=True)
    #
    pred = mx.sym.SwapAxis(data=pred, dim1=0, dim2=1) # (batch_size, seq_len, num_label)

    label = mx.sym.Variable('softmax_label') # (batch_size, seq_len, label_out)
   

 
    #cls_prob = mx.sym.Custom(op_type='FocalLoss', name = 'cls_prob', data = cls_score, labels = label, alpha =0.25, gamma= 2)
    label = mx.sym.Reshape(data=label, shape=(-1,seq_len))
    label = mx.sym.one_hot(label,num_label)
    softmax = mx.sym.softmax(data=pred)
    #val=np.ones((batch_size, seq_len, label_out)).tolist()
    eps=mx.sym.full((32,seq_len,num_label),1e-14)
    eps=mx.sym.BlockGrad(eps)
    softmax = softmax+eps
    cone=mx.sym.full((32,seq_len,num_label),0.5)
    cone = mx.sym.BlockGrad(cone) 
    one=mx.sym.ones((32,seq_len,num_label))
    one =mx.sym.BlockGrad(one)
    softmax_output = mx.sym.BlockGrad(softmax,name = 'softmax')
    #focal
    #loss=mx.sym.broadcast_mul(mx.sym.broadcast_mul(mx.sym.broadcast_mul(mx.sym.pow(one-softmax,1.5),mx.sym.log(softmax)),label),cone) 
    #ce
    loss=mx.sym.broadcast_mul(mx.sym.log(softmax),label)#+mx.sym.broadcast_mul(mx.sym.log(mx.sym.broadcast_sub(one,softmax)),mx.sym.broadcast_sub(one,label))
    ce = -mx.sym.sum(mx.sym.sum(loss,2),0)
    res = mx.symbol.MakeLoss(ce, normalization='batch')
    sm = mx.sym.Group([softmax_output,res])

    #if output_type == 'sigmoid':
    #    sm = mx.sym.LogisticRegressionOutput(data=pred, label=label, name='softmax')
    #else:
    #    label = mx.sym.Reshape(data=label, shape=(-1,seq_len))
    #    sm = mx.sym.SoftmaxOutput(data=pred, label=label, preserve_shape=True, name='softmax')



    init_states = ['l%d_init_c'%l for l in range(num_lstm_layer)]
    init_states += ['l%d_init_h'%l for l in range(num_lstm_layer)]
    
    init_states += ['m_init_m%d'%n for n in range(numM)]
   # init_states += ['one']
    return sm, ('data', 'data_act')+tuple(init_states), ('softmax_label',)

# ##################################
def cnncnnlstm_unroll(num_lstm_layer, seq_len, input_size,
                   num_hidden, num_embed, num_lstm_o, num_label, filter_list, num_filter,
                   max_nbest, max_sentlen, output_type='softmax', dropout=0.):

    embed_weight = mx.sym.Variable("embed_weight")
    embed2_weight = mx.sym.Variable("embed2_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    fc_weight = mx.sym.Variable("fc_weight")
    fc_bias= mx.sym.Variable("fc_bias")
    fc2_weight = mx.sym.Variable("fc2_weight")
    fc2_bias= mx.sym.Variable("fc2_bias")
    fc3_weight = mx.sym.Variable("fc3_weight")
    fc3_bias= mx.sym.Variable("fc3_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    # Step1: construct data cnn model
    cnnparam_cells = []
    for i, filter_size in enumerate(filter_list):
        cnnparam_cells.append(CNNParam(conv_weight=mx.sym.Variable("conv%d_weight" % i),
                                    conv_bias=mx.sym.Variable("conv%d_bias" % i)))
    data = mx.sym.Variable('data') # (batch_size, seq_len, max_nbest, max_sentlen)
    score = mx.sym.Variable('score') # (batch_size, seq_len, max_nbest)
    # embedding layer
    embed = mx.sym.Embedding(data=data, input_dim=input_size, weight=embed_weight,
                             output_dim=num_embed, name='embed') # (batch_size, seq_len, max_nbest, max_sentlen, num_embed)
    conv_input = mx.sym.Reshape(data=embed, shape=(-1, 1, max_sentlen, num_embed)) # (batch_size*seq_len*max_nbest, 1, max_sentlen, num_embed)
    # create convolution + (max) pooling layer for each filter operation
    pooled_outputs = []
    for i, filter_size in enumerate(filter_list):
        convi = mx.sym.Convolution(data=conv_input,
                                   weight=cnnparam_cells[i].conv_weight,
                                   bias=cnnparam_cells[i].conv_bias,
                                   kernel=(filter_size, num_embed),
                                   num_filter=num_filter) # (batch_size*seq_len*max_nbest, num_filter, max_sentlen-filter_size+1, 1)
        relui = mx.sym.Activation(data=convi, act_type='relu')
        pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(max_sentlen - filter_size + 1, 1), stride=(1,1))
        pooled_outputs.append(pooli)
    # combine all pooled outputs
    total_filters = num_filter * len(filter_list)
    Concat = mx.sym.Concat(*pooled_outputs, dim=1)
    h_pool = mx.sym.Reshape(data=Concat, shape=(-1, total_filters)) # (batch_size*seq_len*max_nbest, total_filters)
    # dropout layer
    if dropout > 0.0:
        h_pool = mx.sym.Dropout(data=h_pool, p=dropout)
    score_exp = mx.sym.Reshape(data=score, shape=(-1, 1)) # (batch_size*seq_len*max_nbest, 1)

    # TODO how to deal with nbest information properly ?
    #Concat_data = mx.sym.Concat(h_pool, score_exp, dim=1) # (batch_size*seq_len*max_nbest, total_filters+1)
    Concat_data = mx.sym.broadcast_mul(h_pool, score_exp)
    data_fc = mx.sym.FullyConnected(data=Concat_data, num_hidden=num_hidden, weight=fc1_weight, bias=fc1_bias)
    data_fc = mx.sym.Reshape(data=data_fc, shape=(-1, seq_len, max_nbest, num_hidden))
    data_fc = mx.sym.sum(data=data_fc, axis=2) # (batch_size, seq_len, num_hidden)

    # Step2: construct data_act cnn model
    cnnparam_act_cells = []
    for i, filter_size in enumerate(filter_list):
        cnnparam_act_cells.append(CNNParam(conv_weight=mx.sym.Variable("act_conv%d_weight" % i),
                                    conv_bias=mx.sym.Variable("act_conv%d_bias" % i)))
    data_act = mx.sym.Variable('data_act') # (batch_size, seq_len, max_sentlen)
    # embedding layer
    embed_act = mx.sym.Embedding(data=data_act, input_dim=input_size, weight=embed2_weight,
                             output_dim=num_embed, name='embed_act') # (batch_size, seq_len, max_sentlen, num_embed)
    conv_act_input = mx.sym.Reshape(data=embed_act, shape=(-1, 1, max_sentlen, num_embed)) # (batch_size*seq_len, 1, max_sentlen, num_embed)
    # create convolution + (max) pooling layer for each filter operation
    pooled_act_outputs = []
    for i, filter_size in enumerate(filter_list):
        convi = mx.sym.Convolution(data=conv_act_input,
                                   weight=cnnparam_act_cells[i].conv_weight,
                                   bias=cnnparam_act_cells[i].conv_bias,
                                   kernel=(filter_size, num_embed),
                                   num_filter=num_filter) # (batch_size*seq_len, num_filter, max_sentlen-filter_size+1, 1)
        relui = mx.sym.Activation(data=convi, act_type='relu')
        pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(max_sentlen - filter_size + 1, 1), stride=(1,1))
        pooled_act_outputs.append(pooli)
    # combine all pooled outputs
    total_filters = num_filter * len(filter_list)
    Concat = mx.sym.Concat(*pooled_act_outputs, dim=1)
    h_act_pool = mx.sym.Reshape(data=Concat, shape=(-1, total_filters)) # (batch_size*seq_len, total_filters)
    # dropout layer
    if dropout > 0.0:
        h_act_pool = mx.sym.Dropout(data=h_act_pool, p=dropout)
    data_act_fc = mx.sym.FullyConnected(data=h_act_pool, num_hidden=num_hidden, weight=fc3_weight, bias=fc3_bias)
    data_act_fc = mx.sym.Reshape(data_act_fc, shape=(-1, seq_len, num_hidden))

    # Step3: construct lstm model
    Concat_input = mx.sym.Concat(data_fc, data_act_fc, dim=2, name='Concat') # (batch_size, seq_len, num_hidden*2)
    if dropout > 0.:
        Concat_input = mx.sym.Dropout(data=Concat_input, p=dropout)
    lstm_input = mx.sym.SliceChannel(data=Concat_input, num_outputs=seq_len, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len):
        hidden = lstm_input[seqidx]
        hidden = mx.sym.FullyConnected(data=hidden, num_hidden=num_hidden, weight=fc_weight, bias=fc_bias)
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_lstm_o, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    hidden_Concat = mx.sym.Concat(*hidden_all, dim=0) # (seq_len*batch_size, num_hidden)
    pred = mx.sym.FullyConnected(data=hidden_Concat, num_hidden=num_label, weight=cls_weight, bias=cls_bias, name='pred')
    pred = mx.sym.Reshape(data=pred, shape=(seq_len, -1, num_label))
    pred = mx.sym.SwapAxis(data=pred, dim1=0, dim2=1) # (batch_size, seq_len, num_label)

    label = mx.sym.Variable('softmax_label') # (batch_size, seq_len, label_out)
    if output_type == 'sigmoid':
        sm = mx.sym.LogisticRegressionOutput(data=pred, label=label, name='softmax')
    else:
        label = mx.sym.Reshape(data=label, shape=(-1, seq_len))
        sm = mx.sym.SoftmaxOutput(data=pred, label=label, preserve_shape=True, name='softmax')

    init_states = ['l%d_init_c'%l for l in range(num_lstm_layer)]
    init_states += ['l%d_init_h'%l for l in range(num_lstm_layer)]

    return sm, ('data', 'data_act', 'score')+tuple(init_states), ('softmax_label',)

