from __future__ import division
import tensorflow as tf

import numpy as np
import cPickle as pickle
import os, sys
import scipy.io
import time
from util.rnn import lstm_dynamic_layer as lstm
# from util.rnn_rama import dec_lstm_layer as dec_lstm
from util.cnn import fc_relu_layer as fc_relu
from util.cnn import fc_layer as fc
from util.cnn import conv_layer as conv
from util.bn import batch_norm as bn
from util.custom_init import msr_init
from util import loss as loss_func

class ground_model(object):
    def __init__(self, is_train, config=None):
        self.batch_size = self._init_param(config, 'batch_size', 20)
        self.test_batch_size = self._init_param(config, 'test_batch_size', -1)
        self.class_num = self._init_param(config, 'class_num', 100)
        self.lr = self._init_param(config, 'lr', 0.0001)
        self.init = self._init_param(config, 'init', 'xavier')
        self.optim = self._init_param(config, 'optim', 'adam')
        self.vocab_size = self._init_param(config, 'vocab_size', 17150)
        self.img_feat_size = self._init_param(config, 'img_feat_size', 4096)
        self.dropout = self._init_param(config, 'dropout', 0.7)
        self.num_lstm_layer = self._init_param(config, 'num_lstm_layer', 1)
        self.num_prop = self._init_param(config, 'num_prop', 100)
        self.lstm_dim = self._init_param(config, 'lstm_dim', 1000)
        self.hidden_size = self._init_param(config, 'hidden_size', 128)
        self.phrase_len = self._init_param(config, 'phrase_len', 19)
        self.weight_decay = self._init_param(config, 'weight_decay', 0.0005)

    def _init_param(self, config, param_name, default_value):
        if hasattr(config, param_name):
            return getattr(config, param_name)
        else:
            return default_value

    def init_placeholder(self):
        sen_data = tf.placeholder(tf.int32, [self.batch_size, self.phrase_len])
        enc_data = tf.placeholder(tf.int32, [self.batch_size, self.phrase_len])
        dec_data = tf.placeholder(tf.int32, [self.batch_size, self.phrase_len])
        msk_data = tf.placeholder(tf.int32, [self.batch_size, self.phrase_len])
        vis_data = tf.placeholder(tf.float32, [self.batch_size, self.num_prop, self.img_feat_size])
        bbx_label = tf.placeholder(tf.int32, [self.batch_size])
        is_train = tf.placeholder(tf.bool)
        return sen_data, vis_data, bbx_label, enc_data, dec_data, msk_data, is_train

    def model_structure(self, sen_data, enc_data, dec_data, msk_data, vis_data, batch_size, is_train, dropout=None):
        def set_drop_test():
            return tf.cast(1.0, tf.float32)
        def set_drop_train():
            return tf.cast(self.dropout,tf.float32)
        dropout = tf.cond(is_train,
                          set_drop_train,
                          set_drop_test)

        seq_length = tf.reduce_sum(msk_data, 1)
        text_seq_batch = sen_data
        
        with tf.variable_scope('word_embedding'), tf.device("/cpu:0"):
            embedding_mat = tf.get_variable("embedding", [self.vocab_size, self.lstm_dim], tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))
            # text_seq has shape [T, N] and embedded_seq has shape [T, N, D].
            embedded_seq = tf.nn.embedding_lookup(embedding_mat, text_seq_batch)
            
        # we encode phrase based on the last step of hidden states
        outputs, states = lstm('enc_lstm', embedded_seq, None, seq_length, output_dim=self.lstm_dim,
                        num_layers=1, forget_bias=1.0, apply_dropout=True,keep_prob=dropout,concat_output=False,
                        initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))

        sen_raw = states[-1].h
        sen_raw = tf.nn.l2_normalize(sen_raw, dim=1)

        # print sen_raw.get_shape()
        vis_raw = tf.reshape(vis_data, [self.batch_size*self.num_prop, self.img_feat_size])

        sen_output = tf.reshape(sen_raw, [self.batch_size, 1, 1, self.lstm_dim])
        vis_output = tf.reshape(vis_raw, [self.batch_size, self.num_prop, 1, self.img_feat_size])

        sen_tile = tf.tile(sen_output, [1, self.num_prop, 1, 1])
        feat_concat = tf.concat([sen_tile, vis_output], 3)

        feat_proj_init = msr_init([1, 1, self.lstm_dim+self.img_feat_size, self.hidden_size])
        feat_proj = conv("feat_proj", feat_concat, 1, 1, self.hidden_size, weights_initializer=feat_proj_init)
        feat_relu = tf.nn.relu(feat_proj)

        att_conv_init = msr_init([1, 1, self.hidden_size, 1])
        att_conv = conv("att_conv", feat_relu, 1, 1, 1, weights_initializer=att_conv_init)
        
        #Generate the visual attention feature
        att_scores_t = tf.reshape(att_conv, [self.batch_size, self.num_prop])
        # att_prob = tf.nn.softmax(att_scores_t)
        att_prob = tf.nn.relu(att_scores_t)

        att_scores = tf.reshape(att_prob, [self.batch_size, self.num_prop, 1])

        vis_att_feat = tf.reduce_sum(tf.multiply(vis_data, tf.tile(att_scores, [1,1, self.img_feat_size])), 1)
        vis_att_featFC = fc_relu("vis_enc", vis_att_feat, self.lstm_dim,
        weights_initializer=tf.random_uniform_initializer(minval=-0.002, maxval=0.002)) 

        vis_att_tile = tf.reshape(vis_att_featFC, [self.batch_size, 1, self.lstm_dim])

        text_enc_batch = enc_data
        # embedded_enc: batch_size x phrase_len x lstm_dim
        with tf.variable_scope('enc_embedding'), tf.device("/cpu:0"):
            embedding_enc = tf.get_variable("embedding", [self.vocab_size, self.lstm_dim], tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))
            # text_seq has shape [T, N] and embedded_seq has shape [T, N, D].
            embedded_enc = tf.nn.embedding_lookup(embedding_enc, text_enc_batch)

        # dec_vis_embed = batch_size x phrase_len x (2*lstm_dim)
        dec_vis_embed = tf.concat([embedded_enc, 
            tf.concat([vis_att_tile, tf.zeros((self.batch_size, self.phrase_len-1, self.lstm_dim))], 1)], 2)
        # dec_outputs: batch_size x phrase_len x lstm_dim
        dec_outs, _ = lstm('dec_lstm', dec_vis_embed, None, seq_length, output_dim=self.lstm_dim,
                        num_layers=1, forget_bias=1.0,apply_dropout=True,keep_prob=dropout,concat_output=True,
                        initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08)) 

        dec_outs = tf.reshape(dec_outs, [self.batch_size*self.phrase_len, self.lstm_dim])
        # dec_logits: (batch_size*phrase_len) x vocab_size
        dec_logits = fc('dec_logits', dec_outs, self.vocab_size,
            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True))       

        return att_scores_t, dec_logits, vis_data

    def build_compute_loss(self, dec_logits, target_seqs, input_mask):

        targets = tf.reshape(target_seqs, [-1])
        weights = tf.to_float(tf.reshape(input_mask, [-1]))
        #Compute loss
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=dec_logits)

        batch_loss = tf.reduce_sum(tf.multiply(losses, weights))/tf.reduce_sum(weights)
        total_loss = batch_loss

        weights = self.get_variables_by_name(["weights"], False)
        for cur_w in weights["weights"]:
            total_loss += tf.nn.l2_loss(cur_w)*self.weight_decay

        weight_loss = total_loss-batch_loss
        
        return total_loss, weight_loss, batch_loss

    def get_variables_by_name(self,name_list, verbose=True):
        v_list=tf.trainable_variables()
        v_dict={}
        for name in name_list:
            v_dict[name]=[]
        for v in v_list:
            for name in name_list:
                if name in v.name: v_dict[name].append(v)

        #print
        if verbose:
            for name in name_list:
                print "Variables of <"+name+">"
                for v in v_dict[name]:
                    print "    "+v.name
        return v_dict

    def build_train_op(self, loss):
        if self.optim == 'adam':
            print 'Adam optimizer'
            v_dict = self.get_variables_by_name([""], True)
            var_list1 = [i for i in v_dict[""] if 'vis_enc' not in i.name]
            var_list2 = self.get_variables_by_name(["vis_enc"], True)
            var_list2 = var_list2["vis_enc"]

            opt1 = tf.train.AdamOptimizer(self.lr, name="Adam")
            opt2 = tf.train.AdamOptimizer(self.lr*0.1, name="Adam_vis_enc")
            grads = tf.gradients(loss, var_list1 + var_list2)
            grads1 = grads[:len(var_list1)]
            grads2 = grads[len(var_list1):]
            train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
            train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
            train_op = tf.group(train_op1, train_op2)            
        else:
            print 'SGD optimizer'
            tvars = tf.trainable_variables()
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
            grads = tf.gradients(cost, tvars)
            train_op = optimizer.apply_gradients(zip(grads, tvars))
        return train_op

    def build_eval_op(self, logits):
        softmax_res = tf.nn.softmax(logits)
        return softmax_res

    def build_model(self):
        self.sen_data, self.vis_data, self.bbx_label, self.enc_data, self.dec_data, self.msk_data, self.is_train = self.init_placeholder()
        att_logits, dec_logits, sen_raw = self.model_structure(self.sen_data, self.enc_data, self.dec_data, self.msk_data, self.vis_data, self.batch_size, self.is_train)
        self.total_loss, weight_loss, batch_loss = self.build_compute_loss(dec_logits, self.dec_data, self.msk_data)
        self.train_op = self.build_train_op(self.total_loss)

        return self.total_loss, self.train_op, att_logits, dec_logits




