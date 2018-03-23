import tensorflow as tf
import os, sys
import numpy as np
import time

from dataprovider_supervise import dataprovider
from model_supervise import ground_model
from util.iou import calc_iou
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", type=str, default='grounder')
parser.add_argument("-g", "--gpu", type=str, default='0')
parser.add_argument("--restore_id", type=int, default=0)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

class Config(object):
	batch_size = 40
	img_feat_dir = './feature'
	sen_dir = './annotation'
	train_file_list = 'flickr30k_train_val.lst'
	test_file_list = 'flickr30k_test.lst'
	log_file = './log/ground_supervised'
	save_path = './model/ground_supervised'
	vocab_size = 17150	
	num_epoch = 3
	max_step = 12000
	optim='adam'
	dropout = 0.5
	lr = 0.001
	weight_decay=0.0

def update_feed_dict(dataprovider, model, is_train):
	img_feat, sen_feat, bbx_label = dataprovider.get_next_batch()
	feed_dict = {
				model.sen_data: sen_feat,
				model.vis_data: img_feat,
				model.bbx_label: bbx_label,
				model.is_train: is_train}
	return feed_dict

def eval_cur_batch(gt_label, cur_logits, is_train=True, num_sample=0):
	res_prob = cur_logits
	res_label = np.argmax(res_prob, axis=1)

	accu = 0.0
	if is_train:
		accu = float(np.sum(res_label == gt_label)) / float(len(gt_label))
	else:
		for gt_id, cur_gt in enumerate(gt_label):
			if res_label[gt_id] in cur_gt:
				accu += 1.0

		accu /= float(num_sample)
	return accu

def load_img_id_list(file_list):
	img_list = []
	with open(file_list) as fin:
		for img_id in fin.readlines():
			img_list.append(int(img_id.strip()))
	img_list = np.array(img_list).astype('int')	
	return img_list

def run_eval(sess, dataprovider, model, eval_op, feed_dict):
	num_cnt = 0.0
	num_cor = 0.0
	for img_ind, img_id in enumerate(dataprovider.test_list):
		img_feat_raw, sen_feat_batch, bbx_gt_batch, num_sample_all = dataprovider.get_test_feat(img_id)
		# bbx_gt_batch = set(bbx_gt_batch)

		if num_sample_all > 0:
			num_corr = 0
			num_sample = len(bbx_gt_batch)
			img_feat = feed_dict[model.vis_data]
			for i in range(num_sample):
				img_feat[i] = img_feat_raw
			sen_feat = feed_dict[model.sen_data]
			sen_feat[:num_sample] = sen_feat_batch
			bbx_label = feed_dict[model.bbx_label]

			eval_feed_dict = {
				model.sen_data: sen_feat,
				model.vis_data: img_feat,
				model.bbx_label: bbx_label,
				model.is_train: False}

			cur_att_logits = sess.run(eval_op, feed_dict=eval_feed_dict)
			cur_att_logits = cur_att_logits[:num_sample]
			cur_accuracy = eval_cur_batch(bbx_gt_batch, cur_att_logits, False, num_sample_all)

			print '%d/%d: %d/%d, %.4f'%(img_ind, len(dataprovider.test_list), num_sample, num_sample_all, cur_accuracy)
			num_cor += float(num_sample_all)*cur_accuracy
			num_cnt += float(num_sample_all)
		else:
			print 'No gt for %d'%img_id

	accu = num_cor/num_cnt
	print 'Accuracy = %.4f'%accu
	return accu

def run_evaluate():
	train_list = []
	test_list = []
	config = Config()
	train_list = load_img_id_list(config.train_file_list)
	test_list = load_img_id_list(config.test_file_list)

	config.save_path = config.save_path + '_' + args.model_name
	assert(os.path.isdir(config.save_path))
	restore_id = args.restore_id
	assert(restore_id > 0)

	cur_dataset = dataprovider(train_list, test_list, config.img_feat_dir, config.sen_dir, config.vocab_size,
								batch_size=config.batch_size)

	model = ground_model(config)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

	with tf.Graph().as_default():
		loss, train_op, loss_vec, logits = model.build_model()
		# Create a session for running Ops on the Graph.
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		# Run the Op to initialize the variables.
		init = tf.global_variables_initializer()
		sess.run(init)
		saver = tf.train.Saver(max_to_keep=100)

		feed_dict = update_feed_dict(cur_dataset, model, True)
		saver.restore(sess, './model/%s/model_%d.ckpt'%(config.save_path, restore_id)) 
				
		print "-----------------------------------------------"
		eval_accu = run_eval(sess, cur_dataset, model, logits, feed_dict)
		print "-----------------------------------------------"


def main(_):
	run_evaluate()

if __name__ == '__main__':
    tf.app.run()