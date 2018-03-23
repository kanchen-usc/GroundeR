from __future__ import division
import tensorflow as tf

import numpy as np
import cPickle as pickle
import os, sys
import scipy.io
import time
from util.rnn import lstm_layer as lstm
from util.cnn import fc_relu_layer as fc_relu
from util.cnn import fc_layer as fc
from util.cnn import conv_layer as conv
from util.bn import batch_norm as bn
from util.custom_init import msr_init
from util import loss as loss_func

class ground_model(object):
	def __init__(self, config=None):
		self.batch_size = self._init_param(config, 'batch_size', 40)
		self.test_batch_size = self._init_param(config, 'test_batch_size', -1)
		self.class_num = self._init_param(config, 'class_num', 100)
		self.lr = self._init_param(config, 'lr', 0.0001)
		self.init = self._init_param(config, 'init', 'axvier')
		self.optim = self._init_param(config, 'optim', 'adam')
		self.vocab_size = self._init_param(config, 'vocab_size', 17150)
		self.img_feat_size = self._init_param(config, 'img_feat_size', 4096)
		self.dropout = self._init_param(config, 'dropout', 0.5)
		self.num_lstm_layer = self._init_param(config, 'num_lstm_layer', 1)
		self.num_prop = self._init_param(config, 'num_prop', 100)
		self.lstm_dim = self._init_param(config, 'lstm_dim', 500)
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
		vis_data = tf.placeholder(tf.float32, [self.batch_size, self.num_prop, self.img_feat_size])
		bbx_label = tf.placeholder(tf.int32, [self.batch_size])
		is_train = tf.placeholder(tf.bool)
		return sen_data, vis_data, bbx_label, is_train

	def model_structure(self, sen_data, vis_data, batch_size, is_train, dropout=None):
		if dropout == None:
			dropout = self.dropout

		text_seq_batch = tf.transpose(sen_data, [1, 0])	# input data is [num_steps, batch_size]
		with tf.variable_scope('word_embedding'), tf.device("/cpu:0"):
			embedding_mat = tf.get_variable("embedding", [self.vocab_size, self.lstm_dim], tf.float32,
				initializer=tf.contrib.layers.xavier_initializer(uniform=True))
			# text_seq has shape [T, N] and embedded_seq has shape [T, N, D].
			embedded_seq = tf.nn.embedding_lookup(embedding_mat, text_seq_batch)
		# we encode phrase based on the last step of hidden states
		_, states = lstm('lstm_lang', embedded_seq, None, output_dim=self.lstm_dim,
						num_layers=1, forget_bias=1.0, apply_dropout=False,concat_output=False,
						initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))

		# batch normalization for visual and language part
		sen_raw = states[-1].h
		vis_raw = tf.reshape(vis_data, [self.batch_size*self.num_prop, self.img_feat_size])

		sen_bn = bn(sen_raw, is_train, "SEN_BN", 0.9)
		vis_bn = bn(vis_raw, is_train, "VIS_BN", 0.9)

		sen_output = tf.reshape(sen_bn, [self.batch_size, 1, 1, self.lstm_dim])
		vis_output = tf.reshape(vis_bn, [self.batch_size, self.num_prop, 1, self.img_feat_size])

		sen_tile = tf.tile(sen_output, [1, self.num_prop, 1, 1])
		feat_concat = tf.concat([sen_tile, vis_output], 3)

		feat_proj_init = msr_init([1, 1, self.lstm_dim+self.img_feat_size, self.hidden_size])
		feat_proj = conv("feat_proj", feat_concat, 1, 1, self.hidden_size, weights_initializer=feat_proj_init)
		feat_relu = tf.nn.relu(feat_proj)

		att_conv_init = msr_init([1, 1, self.hidden_size, 1])
		att_conv = conv("att_conv", feat_relu, 1, 1, 1, weights_initializer=att_conv_init)
		att_scores = tf.reshape(att_conv, [self.batch_size, self.num_prop])

		return att_scores

	def build_compute_loss(self, att_logits, labels):
		loss_vec=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=att_logits, labels=labels, name=None)
		loss_cls = tf.reduce_mean(loss_vec)
		reg_var_list = self.get_variables_by_name([""], False)
		loss_reg = loss_func.l2_regularization_loss(reg_var_list[""], self.weight_decay)
		loss = loss_cls + loss_reg
		return loss, loss_vec

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
			v_dict = self.get_variables_by_name([""])
			optimizer = tf.train.AdamOptimizer(self.lr,name='Adam')
			train_op = optimizer.minimize(loss, var_list=v_dict[""])
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
		self.sen_data, self.vis_data, self.bbx_label, self.is_train = self.init_placeholder()
		att_logits = self.model_structure(self.sen_data, self.vis_data, self.batch_size, self.is_train)
		self.loss, loss_vec = self.build_compute_loss(att_logits, self.bbx_label)
		self.train_op = self.build_train_op(self.loss)

		return self.loss, self.train_op, loss_vec, att_logits




