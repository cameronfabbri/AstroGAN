'''

   This file trains a model to predict the attributes given an image. For training, it uses
   either the true train data, or the generated train data.

'''

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import scipy.misc as misc
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import cPickle as pickle
from tqdm import tqdm
import numpy as np
import argparse
import random
import ntpath
import time
import glob
import sys
import cv2
import os
from sklearn.metrics import mean_squared_error
from math import sqrt

sys.path.insert(0, '../../ops/')
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

from tf_ops import *
from nets import *
import data_ops
from config import classes
import inception_resnet_v2
import alexnet
slim = tf.contrib.slim

'''
   Loads the data specified, either generated or real, and also with/without redshift
'''
def loadData(data_dir, data_type, use_both, classes):

   if data_type == 'real':
      train_paths = sorted(glob.glob(data_dir+'images_training_rev1/train/*.jpg'))
      train_ids   = [ntpath.basename(x.split('.')[0]) for x in train_paths]
      
      test_paths = sorted(glob.glob(data_dir+'images_training_rev1/test/*.jpg'))
      test_ids   = [ntpath.basename(x.split('.')[0]) for x in test_paths]

      train_attributes = []
      test_attributes  = []

      d = 0
      with open(data_dir+'training_solutions_rev1.csv', 'r') as f:
         for line in f:
            if d == 0:
               d = 1
               continue
            line = np.asarray(line.split(',')).astype('float32')
            im_id = int(line[0])
            att = line[1:]

            # remember train_ids is all str
            if str(im_id) in train_ids:
               train_attributes.append(att)
            else:
               test_attributes.append(att)

   # going to use both real data and generated data
   if data_type == 'gen':
      print 'using gen data'

      pkl_file = open(data_dir+'data.pkl', 'rb')
      data_info = pickle.load(pkl_file)

      train_paths = sorted(glob.glob(data_dir+'*.png'))
      train_ids   = [ntpath.basename(x.split('.')[0]) for x in train_paths]

      test_paths = sorted(glob.glob('/mnt/data1/images/galaxyzoo/images_training_rev1/test/*.jpg'))
      test_ids   = [ntpath.basename(x.split('.')[0]) for x in test_paths]
      
      train_attributes = []
      test_attributes  = []

      # getting the attributes for generated data.
      for tid in train_ids:
         train_attributes.append(np.squeeze(data_info[tid+'.png']))
      d = 0
      with open('/mnt/data1/images/galaxyzoo/training_solutions_rev1.csv', 'r') as f:
         for line in f:
            if d == 0:
               d = 1
               continue
            line = np.asarray(line.split(',')).astype('float32')
            im_id = int(line[0])
            att = line[1:]

            # remember ids is all str
            if str(im_id) in test_ids:
               test_attributes.append(att)

      if use_both == True:
         print 'Using real data along with gen'
         # real data loading. Repetitive, but works
         train_paths = train_paths + sorted(glob.glob('/mnt/data1/images/galaxyzoo/images_training_rev1/train/*.jpg'))
         train_ids   = train_ids + [ntpath.basename(x.split('.')[0]) for x in train_paths]
         
         d = 0
         with open('/mnt/data1/images/galaxyzoo/training_solutions_rev1.csv', 'r') as f:
            for line in f:
               if d == 0:
                  d = 1
                  continue
               line = np.asarray(line.split(',')).astype('float32')
               im_id = int(line[0])
               att = line[1:]
               # remember train_ids is all str
               if str(im_id) in train_ids:
                  train_attributes.append(att)

   train_paths = np.asarray(train_paths)
   train_attributes = np.asarray(train_attributes)
   train_ids = np.asarray(train_ids)
   test_paths = np.asarray(test_paths)
   test_attributes = np.asarray(test_attributes)
   test_ids = np.asarray(test_ids)
   return train_paths, train_attributes, train_ids, test_paths, test_attributes, test_ids

if __name__ == '__main__':

   SIZE = 224

   parser = argparse.ArgumentParser()
   parser.add_argument('--BATCH_SIZE', required=False,help='Batch size', type=int,default=64)
   parser.add_argument('--DATA_TYPE',  required=True,help='Real or generated data',type=str)
   parser.add_argument('--DATA_DIR',   required=True,help='Data directory',type=str)
   parser.add_argument('--NETWORK',    required=True,help='Which network',type=str)
   parser.add_argument('--USE_BOTH',   required=True,help='Use both real and gen',type=int)
   parser.add_argument('--EPOCHS',     required=False,help='Number of epochs',type=int,default=100)
   a = parser.parse_args()

   BATCH_SIZE     = a.BATCH_SIZE
   DATA_TYPE      = a.DATA_TYPE
   DATA_DIR       = a.DATA_DIR
   NETWORK        = a.NETWORK
   EPOCHS         = a.EPOCHS
   use_both       = bool(a.USE_BOTH)

   # using both only applies to when using gen
   if DATA_TYPE == 'real': use_both = bool(0)

   CHECKPOINT_DIR = 'checkpoints/'+'DATA_TYPE_'+DATA_TYPE+'/NETWORK_'+NETWORK+'/USE_BOTH_'+str(use_both)+'/'
   try: os.makedirs(CHECKPOINT_DIR)
   except: pass
   
   info_dict = {}
   info_dict['BATCH_SIZE'] = BATCH_SIZE
   info_dict['DATA_TYPE']  = DATA_TYPE
   info_dict['DATA_DIR']   = DATA_DIR
   info_dict['USE_BOTH']   = use_both
   info_dict['NETWORK']    = NETWORK
   info_dict['EPOCHS']     = EPOCHS
   exp_pkl = open(CHECKPOINT_DIR+'info.pkl', 'wb')
   data = pickle.dumps(info_dict)
   exp_pkl.write(data)
   exp_pkl.close()

   global_step = tf.Variable(0, name='global_step', trainable=False)
   images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, SIZE, SIZE, 3), name='real_images')
   labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 37), name='attributes')

   LR = tf.placeholder(tf.float32, name='learning_rate')

   # clip logits between [0, 1] because that's the range of the labels
   if NETWORK == 'inception':
      print 'Using inception'
      with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
         logits, _ = inception_resnet_v2.inception_resnet_v2(images, num_classes=37, is_training=True)
         logits = tf.minimum(tf.maximum(0.0,logits), 1.0)
   if NETWORK == 'alexnet':
      print 'Using alexnet'
      with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
         logits, _ = alexnet.alexnet_v2(images, num_classes=37, is_training=True)
         logits = tf.minimum(tf.maximum(0.0,logits), 1.0)
   print 'Done.'

   loss = tf.reduce_mean(tf.nn.l2_loss(logits-labels))

   tf.summary.scalar('loss', loss)
   summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR+'/'+'logs/', graph=tf.get_default_graph())
   merged_summary_op = tf.summary.merge_all()
   
   train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss, global_step=global_step)
   
   saver = tf.train.Saver(max_to_keep=1)
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess  = tf.Session()
   sess.run(init)
   # restore previous model if there is one
   ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print 'Restoring previous model...'
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print 'Model restored'
      except:
         print 'Could not restore model'
         pass

   print 'Loading data...'
   train_images, train_annots, train_ids, test_images, test_annots, test_ids = loadData(DATA_DIR, DATA_TYPE, use_both, classes)
   print 'Done.'

   train_len = len(train_annots)
   test_len  = len(test_annots)
   step      = sess.run(global_step)

   epoch_num = step/(train_len/BATCH_SIZE)

   while epoch_num < EPOCHS:

      epoch_num = step/(train_len/BATCH_SIZE)

      idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
      batch_y      = train_annots[idx]
      batch_paths  = train_images[idx]

      # create batch images
      batch_images = np.empty((BATCH_SIZE, SIZE, SIZE, 3), dtype=np.float32)

      i = 0
      learning_rate = 1e-4
      for p in batch_paths:
         img = misc.imread(p).astype('float32')
         img = misc.imresize(img, (SIZE,SIZE))
         #img = img/255.0
         batch_images[i, ...] = img
         i += 1
         
      # randomly flip batch
      r = random.random()
      # flip image left right
      if r < 0.5:
         batch_images = np.fliplr(batch_images)
      r = random.random()
      # flip image up down
      if r < 0.5:
         batch_images = np.flipud(batch_images)

      _, loss_, summary = sess.run([train_op, loss, merged_summary_op], feed_dict={images:batch_images, labels:batch_y, LR:learning_rate})
      
      summary_writer.add_summary(summary, step)

      print 'epoch:',epoch_num,'step:',step,'loss:',loss_

      step += 1

      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')

         idx          = np.random.choice(np.arange(test_len), BATCH_SIZE, replace=False)
         batch_y      = test_annots[idx]
         batch_paths  = test_images[idx]
      
         i = 0
         for p in batch_paths:
            img = misc.imread(p).astype('float32')
            img = misc.imresize(img, (SIZE,SIZE))
            #img = img/255.0
            batch_images[i, ...] = img
            i += 1

         preds = np.asarray(sess.run([logits], feed_dict={images:batch_images, labels:batch_y}))[0]

         f = open(CHECKPOINT_DIR+'testing.txt', 'a')
         batch_err = 0
         for r,p in zip(batch_y, preds):
            # root mean squared error
            batch_err = batch_err+sqrt(mean_squared_error(r, p))
         batch_err = float(batch_err)/float(BATCH_SIZE)
         f.write(str(step)+','+str(batch_err)+'\n')
         f.close()

   print 'Training done'
