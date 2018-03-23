import tensorflow as tf
import os, sys
import numpy as np
import time

#Provides data in batches for training and reads test feat for test
from dataprovider_unsupervise import dataprovider
#Provides the model and loss functions for grouding algo
from model_unsupervise import ground_model
#Utility functions to compute iou
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
    log_file = './log/ground_unsupervised'
    save_path = './model/ground_unsupervised'
    vocab_size = 17150    
    num_epoch = 3
    max_step = 100000
    optim='adam'
    dropout = 0.5
    lr = 0.0001
    phrase_len=5
    weight_decay=0.0005
    lstm_dim = 500

#Generate the input data feed batches for train and test
def update_feed_dict(dataprovider, model, is_train):
    img_feat, sen_feat, enc_batch, dec_batch, mask_batch, bbx_label = dataprovider.get_next_batch()
    feed_dict = {
                model.sen_data: sen_feat,
                model.vis_data: img_feat,
                model.bbx_label: bbx_label,
                model.enc_data: enc_batch,
                model.dec_data: dec_batch,
                model.msk_data: mask_batch,
                model.is_train: is_train}
    return feed_dict

#Evaluate the current batch of samples
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

#Function to load train and test image lists
def load_img_id_list(file_list):
    img_list = []
    with open(file_list) as fin:
        for img_id in fin.readlines():
            img_list.append(int(img_id.strip()))
    img_list = np.array(img_list).astype('int')    
    return img_list

def run_eval(sess, dataprovider, model, eval_op, feed_dict):
    accu = 0.0
    num_cnt = 0.0
    for img_ind, img_id in enumerate(dataprovider.test_list):
        img_feat_raw, sen_feat_batch, mask_batch, bbx_gt_batch, num_sample_all = dataprovider.get_test_feat(img_id)
        # print sen_feat_batch

        if num_sample_all > 0:
            num_corr = 0
            num_sample = len(bbx_gt_batch)
            img_feat = feed_dict[model.vis_data]
            for i in range(num_sample):
                img_feat[i] = img_feat_raw
            sen_feat = feed_dict[model.sen_data]
            sen_feat[:num_sample] = sen_feat_batch
            mask_data = feed_dict[model.msk_data]
            mask_data[:num_sample] = mask_batch

            eval_feed_dict = {
                model.sen_data: sen_feat,
                model.vis_data: img_feat,
                model.msk_data: mask_data,
                model.is_train: False}

            cur_att_logits = sess.run(eval_op, feed_dict=eval_feed_dict)
            cur_att_logits = cur_att_logits[:num_sample]
            cur_accuracy = eval_cur_batch(bbx_gt_batch, cur_att_logits, False, num_sample_all)

            print '%d/%d: %d/%d, %.4f'%(img_ind, len(dataprovider.test_list), num_sample, num_sample_all, cur_accuracy)
            accu += cur_accuracy*num_sample_all
            num_cnt += float(num_sample_all)
        else:
            print 'No gt for %d'%img_id

    accu /= num_cnt
    print 'Accuracy = %.4f'%(accu)
    return accu

#The main function that runs training
def run_training():
    train_list = []
    test_list = []
    config = Config()
    train_list = load_img_id_list(config.train_file_list)
    test_list = load_img_id_list(config.test_file_list)

    #Directory to save model Info
    config.save_path = config.save_path + '_' + args.model_name
    if not os.path.isdir(config.save_path):
        print 'Save models into %s'%config.save_path
        os.mkdir(config.save_path)
    #Log File
    log_file = config.log_file + '_' + args.model_name + '.log'
    restore_id = args.restore_id
    if restore_id > 0:
        log_file = config.log_file + '_' + args.model_name + '_restore_%d.log'%restore_id
    log = open(log_file, 'w', 0)

    #Initialize the paths and parameters for the current dataset
    cur_dataset = dataprovider(train_list, test_list, config.img_feat_dir, config.sen_dir, config.vocab_size,
                                phrase_len=config.phrase_len, batch_size=config.batch_size)
    is_train = True
    #Initialize groundeR model train instance
    model = ground_model(is_train, config)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

    with tf.Graph().as_default():
        #Build the model
        total_loss, train_op, att_logits, dec_logits = model.build_model()
        # Create a session for running Ops on the Graph.
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # Run the Op to initialize the variables.
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=100)
        duration = 0.0

        if restore_id > 0:
            print 'Restore model_%d'%restore_id
            cur_dataset.epoch_id = restore_id
            cur_dataset.is_save = False
            saver.restore(sess, '%s/model_%d.ckpt'%(config.save_path, restore_id))

        for step in xrange(config.max_step):
            start_time = time.time()
            feed_dict = update_feed_dict(cur_dataset, model, is_train)

            _, cur_tot_loss, cur_logits = sess.run([train_op, total_loss, att_logits], feed_dict=feed_dict)
            duration += time.time()-start_time

            if cur_dataset.is_save:
                print 'Save model_%d into %s'%(cur_dataset.epoch_id, config.save_path)
                saver.save(sess, '%s/model_%d.ckpt'%(config.save_path, cur_dataset.epoch_id))
                cur_dataset.is_save = False

            if step%10 == 0:
                cur_accu = eval_cur_batch(feed_dict[model.bbx_label], cur_logits, is_train=is_train)
                print 'Step %d: loss = %.4f, accu = %.4f (%.4f sec)'%(step, cur_tot_loss, cur_accu, duration/10.0)
                duration = 0.0

            if (step%600)==0:
                print "-----------------------------------------------"
                eval_accu = run_eval(sess, cur_dataset, model, att_logits, feed_dict)
                log.write('%d/%d: %.4f, %.4f\n'%(step+1, cur_dataset.epoch_id, cur_tot_loss, eval_accu))
                print "-----------------------------------------------"
                model.batch_size = config.batch_size
                cur_dataset.is_save = False

    log.close()

def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()
