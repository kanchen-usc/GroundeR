from __future__ import division

import numpy as np
import cPickle as pickle
import os, sys
import scipy.io

class dataprovider(object):
	def __init__(self, train_list, test_list, img_feat_dir, sen_dir, vocab_size,
		val_list='', phrase_len=19, batch_size=40, seed=1):
		self.train_list = train_list
		self.val_list = val_list
		self.test_list = test_list
		self.img_feat_dir = img_feat_dir
		self.sen_dir = sen_dir
		self.phrase_len = phrase_len
		self.cur_id = 0
		self.epoch_id = 0
		self.num_prop = 100
		self.img_feat_size = 4096
		self.num_test = 1000
		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.is_save = False
		np.random.seed(seed)
		self.train_id_list = np.random.permutation(len(train_list))

	def _reset(self):
		self.cur_id = 0
		self.train_id_list = np.random.permutation(len(self.train_list))
		self.is_save = False

	def _read_single_feat(self, img_id):
		sen_feat = np.load('%s/%d.pkl'%(self.sen_dir, img_id))
		pos_ids = np.array(sen_feat['pos_id']).astype('int')
		pos_ind = np.where(pos_ids != -1)[0]

		if len(pos_ind) > 0:
			img_feat = np.zeros((self.num_prop, self.img_feat_size))
			# print '%s/%d.npy'%(self.img_feat_dir, img_id)
			cur_feat = np.load('%s/%d.npy'%(self.img_feat_dir, img_id))
			img_feat[:cur_feat.shape[0], :] = cur_feat
			img_feat = img_feat.astype('float')

			sens = sen_feat['sens']
			sen_id = np.random.randint(len(pos_ind))
			# print img_id, sen_id
			sen = sens[pos_ind[sen_id]]
			# pad sen tokens to phrase_len with UNK token as (self.vocab_size-1)
			sen_token = np.ones(self.phrase_len)*(self.vocab_size-1)	
			sen_token = sen_token.astype('int')
			sen_token[:len(sen)] = sen
			y = pos_ids[pos_ind[sen_id]]
			return img_feat, sen_token, y
		else:
			return None, None, -1

	def get_next_batch(self):
		img_feat_batch = np.zeros((self.batch_size, self.num_prop, self.img_feat_size)).astype('float')
		token_batch = np.zeros((self.batch_size, self.phrase_len)).astype('int')
		y_batch = np.zeros(self.batch_size).astype('int')
		num_cnt = 0
		while num_cnt < self.batch_size:
			if self.cur_id == len(self.train_list):
				self._reset()
				self.epoch_id += 1
				self.is_save = True
				print('Epoch %d complete'%(self.epoch_id))
			img_id = self.train_list[self.train_id_list[self.cur_id]]		
			img_feat, sen_token, y = self._read_single_feat(img_id)
			if y != -1:
				img_feat_batch[num_cnt] = img_feat
				token_batch[num_cnt] = sen_token
				y_batch[num_cnt] = y
				num_cnt += 1
			# else:
			# 	print('No positive samples for %d'%(self.train_list[self.train_id_list[self.cur_id]]))
			self.cur_id += 1		
		return img_feat_batch, token_batch, y_batch

	def get_test_feat(self, img_id):
		sen_feat = np.load('%s/%d.pkl'%(self.sen_dir, img_id))
		pos_ids = np.array(sen_feat['pos_id']).astype('int')
		pos_ind = np.where(pos_ids != -1)[0]
		gt_pos_all = sen_feat['gt_pos_all']
		num_sample = len(pos_ids)

		if len(pos_ind) > 0:
			img_feat = np.zeros((self.num_prop, self.img_feat_size)).astype('float')
			cur_feat = np.load('%s/%d.npy'%(self.img_feat_dir, img_id)).astype('float')
			img_feat[:cur_feat.shape[0], :] = cur_feat
			sen_feat_batch = np.zeros((len(pos_ind), self.phrase_len)).astype('int')
			# y_batch = np.zeros(len(pos_ind)).astype('int')
			gt_batch = []

			sens = sen_feat['sens']
			for sen_ind in range(len(pos_ind)):
				cur_sen = sens[pos_ind[sen_ind]]
				sen_token = np.ones(self.phrase_len)*(self.vocab_size-1)
				sen_token = sen_token.astype('int')
				sen_token[:len(cur_sen)] = cur_sen
				sen_feat_batch[sen_ind] = sen_token
				gt_batch.append(gt_pos_all[pos_ind[sen_ind]])

			# print y_batch
			return img_feat, sen_feat_batch, gt_batch, num_sample
		else:
			return None, None, None, 0

if __name__ == '__main__':
	train_list = []
	test_list = []
	img_feat_dir = '~/dataset/flickr30k_img_bbx_ss_vgg_cls'
	sen_dir = '~/dataset/flickr30k_img_sen_feat'
	vocab_size = 17150
	with open('flickr30k_train.lst') as fin:
		for img_id in fin.readlines():
			train_list.append(int(img_id.strip()))
	train_list = np.array(train_list).astype('int')
	cur_dataset = dataprovider(train_list, test_list, img_feat_dir, sen_dir, vocab_size)
	for i in range(10000):
		img_feat_batch, token_batch, y_batch = cur_dataset.get_next_batch()
		print img_feat_batch.shape
		print y_batch
		print '%d/%d'%(cur_dataset.cur_id, len(cur_dataset.train_list))





