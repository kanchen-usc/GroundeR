from __future__ import division

import numpy as np
import cPickle as pickle
import os, sys
import scipy.io

class dataprovider(object):
    def __init__(self, train_list, test_list, img_feat_dir, sen_dir, vocab_size,
        val_list='', phrase_len=5, batch_size=20, seed=1):
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
        # img_id = self.train_list[self.train_id_list[self.cur_id]]

        sen_feat = np.load('%s/%d.pkl'%(self.sen_dir, img_id))
        pos_ids = np.array(sen_feat['pos_id']).astype('int')
        pos_ind = np.where(pos_ids != -1)[0]

        if len(pos_ind) > 0:
            img_feat = np.zeros((self.num_prop, self.img_feat_size))
            cur_feat = np.load('%s/%d.npy'%(self.img_feat_dir, img_id))

            cur_feat_norm = np.sqrt((cur_feat*cur_feat).sum(axis=1))
            cur_feat /= cur_feat_norm.reshape(cur_feat.shape[0], 1)

            img_feat[:cur_feat.shape[0], :] = cur_feat
            img_feat = img_feat.astype('float')

            sens = sen_feat['sens']
            sen_id = np.random.randint(len(pos_ind))
            # print img_id, sen_id
            sen = sens[pos_ind[sen_id]]
            if len(sen) > self.phrase_len:
                sen = sen[:self.phrase_len]

            # pad sen tokens to phrase_len with UNK token as (self.vocab_size-1)
            sen_token = np.ones(self.phrase_len, dtype=int)*(self.vocab_size-1)    
            enc_token = np.ones(self.phrase_len, dtype=int)*(self.vocab_size-1)    
            dec_token = np.ones(self.phrase_len, dtype=int)*(self.vocab_size-1)    
            indicator = np.zeros(self.phrase_len, dtype=int)
            sen_token[:len(sen)] = sen
            enc_token[:] = sen_token
            dec_token[:-1] = enc_token[1:]

            indicator[:len(sen)] = 1
            y = pos_ids[pos_ind[sen_id]]
            return img_feat, sen_token, enc_token, dec_token, indicator, y
        else:
            return None, None, None, None, None, -1

    def get_next_batch(self):
        img_feat_batch = np.zeros((self.batch_size, self.num_prop, self.img_feat_size)).astype('float')
        token_batch = np.zeros((self.batch_size, self.phrase_len)).astype('int')
        enc_batch = np.zeros((self.batch_size, self.phrase_len)).astype('int')
        dec_batch = np.zeros((self.batch_size, self.phrase_len)).astype('int')
        mask_batch = np.zeros((self.batch_size, self.phrase_len)).astype('int')
        y_batch = np.zeros(self.batch_size).astype('int')
        num_cnt = 0
        while num_cnt < self.batch_size:
            if self.cur_id == len(self.train_list):
                self._reset()
                self.epoch_id += 1
                self.is_save = True
                print('Epoch %d complete'%(self.epoch_id))
            img_id = self.train_list[self.train_id_list[self.cur_id]]        
            img_feat, sen_token, enc_token, dec_token, indicator, y = self._read_single_feat(img_id)
            if y != -1:
                img_feat_batch[num_cnt] = img_feat
                token_batch[num_cnt] = sen_token
                y_batch[num_cnt] = y
                enc_batch[num_cnt] = enc_token
                dec_batch[num_cnt] = dec_token
                mask_batch[num_cnt] = indicator
                num_cnt += 1
            # else:
            #     print('No positive samples for %d'%(self.train_list[self.train_id_list[self.cur_id]]))
            self.cur_id += 1   
        return img_feat_batch, token_batch, enc_batch, dec_batch, mask_batch, y_batch

    def get_test_feat(self, img_id):
        sen_feat = np.load('%s/%d.pkl'%(self.sen_dir, img_id))
        pos_ids = np.array(sen_feat['pos_id']).astype('int')
        pos_ind = np.where(pos_ids != -1)[0]
        gt_pos_all = sen_feat['gt_pos_all']
        gt_bbx_all = sen_feat['gt_box']     # ground truth bbx for query: [xmin, ymin, xmax, ymax]
        num_sample = len(pos_ids)
        num_corr = 0

        if len(pos_ids) > 0:
            img_feat = np.zeros((self.num_prop, self.img_feat_size)).astype('float')
            cur_feat = np.load('%s/%d.npy'%(self.img_feat_dir, img_id)).astype('float')

            cur_feat_norm = np.sqrt((cur_feat*cur_feat).sum(axis=1))
            cur_feat /= cur_feat_norm.reshape(cur_feat.shape[0], 1)
            
            img_feat[:cur_feat.shape[0], :] = cur_feat
            sen_feat_batch = np.zeros((len(pos_ind), self.phrase_len)).astype('int')
            mask_batch = np.zeros((len(pos_ind), self.phrase_len)).astype('int')
            gt_batch = []

            sens = sen_feat['sens']
            for sen_ind in range(len(pos_ind)):
                cur_sen = sens[pos_ind[sen_ind]]
                sen_token = np.ones(self.phrase_len)*(self.vocab_size-1)
                sen_token = sen_token.astype('int')
                if len(cur_sen) > self.phrase_len:
                    cur_sen = cur_sen[:self.phrase_len]
                sen_token[:len(cur_sen)] = cur_sen
                sen_feat_batch[sen_ind] = sen_token
                mask_batch[sen_ind][:len(cur_sen)] = 1
                gt_batch.append(gt_pos_all[pos_ind[sen_ind]])

            for sen_ind in range(len(pos_ids)):
                if not np.any(gt_bbx_all[sen_ind]):
                    num_sample -= 1

            return img_feat, sen_feat_batch, mask_batch, gt_batch, num_sample
        else:
            return None, None, None, None, 0

if __name__ == '__main__':
    train_list = []
    test_list = []
    img_feat_dir = '~/dataset/flickr30k_img_bbx_ss_vgg_det'
    sen_dir = '~/dataset/flickr30k_img_sen_feat'
    vocab_size = 17150
    with open('../flickr30k_test.lst') as fin:
        for img_id in fin.readlines():
            train_list.append(int(img_id.strip()))
    train_list = np.array(train_list).astype('int')
    cur_dataset = dataprovider(train_list, test_list, img_feat_dir, sen_dir, vocab_size)
    for i in range(10000):
        img_feat_batch, token_batch, enc_batch, dec_batch, mask_batch, y_batch = cur_dataset.get_next_batch()
        # img_feat_batch, sen_feat_batch, mask_batch, gt_batch, num_sample = cur_dataset.get_test_feat(train_list[cur_dataset.cur_id])
        print img_feat_batch.shape#, token_batch.shape, enc_batch.shape, dec_batch.shape, mask_batch.shape 
        # print y_batch
        print token_batch
        print token_batch[:,1]#, enc_batch[:,1], dec_batch[:,1], mask_batch[:,1] 
        print '%d/%d'%(cur_dataset.cur_id, len(cur_dataset.train_list))
