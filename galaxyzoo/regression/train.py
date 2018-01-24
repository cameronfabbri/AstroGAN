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

sys.path.insert(0, '../../ops/')
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

from tf_ops import *
from nets import *
import data_ops
from config import classes
import inception_resnet_v2
slim = tf.contrib.slim
import cifarnet

'''
   Loads the data specified, either generated or real, and also with/without redshift
'''
def loadData(data_dir, data_type, classes):

   # data must be of size 299
   if data_type == 'real':
      train_paths = sorted(glob.glob(data_dir+'images_training_rev1/train/*.jpg'))
      train_ids   = [ntpath.basename(x.split('.')[0]) for x in train_paths]
      
      test_paths = sorted(glob.glob(data_dir+'images_training_rev1/test/*.jpg'))
      test_ids   = [ntpath.basename(x.split('.')[0]) for x in test_paths]

      # put ids with image path just to be sure we get the correct image. Slow but oh well, only done once
      #id_dict = {}
      #for img, id_ in zip(train_paths, train_ids):
      #   id_dict[int(id_)] = img
      #for img, id_ in zip(test_paths, test_ids):
      #   id_dict[int(id_)] = img

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

      train_paths = np.asarray(train_paths)
      train_attributes = np.asarray(train_attributes)
      train_ids = np.asarray(train_ids)
      test_paths = np.asarray(test_paths)
      test_attributes = np.asarray(test_attributes)
      test_ids = np.asarray(test_ids)

      return train_paths, train_attributes, train_ids, test_paths, test_attributes, test_ids

   elif data_type == 'gen':
      print 'using gen data'


if __name__ == '__main__':
         
   parser = argparse.ArgumentParser()
   parser.add_argument('--BATCH_SIZE', required=False,help='Batch size', type=int,default=64)
   parser.add_argument('--DATA_TYPE',  required=True,help='Real or generated data',type=str)
   parser.add_argument('--DATA_DIR',   required=True,help='Data directory',type=str)
   parser.add_argument('--EPOCHS',     required=False,help='Number of epochs',type=int,default=100)
   a = parser.parse_args()

   BATCH_SIZE     = a.BATCH_SIZE
   DATA_TYPE      = a.DATA_TYPE
   DATA_DIR       = a.DATA_DIR
   EPOCHS         = a.EPOCHS

   CHECKPOINT_DIR = 'checkpoints/'+'DATA_TYPE_'+DATA_TYPE+'/'
   try: os.makedirs(CHECKPOINT_DIR)
   except: pass

   global_step = tf.Variable(0, name='global_step', trainable=False)
   images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 299, 299, 3), name='real_images')
   labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 37), name='attributes')

   print 'Loading inception resnet v2...'
   with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
      logits, _ = inception_resnet_v2.inception_resnet_v2(images, num_classes=37, is_training=True)
   print 'Done.'

   loss = tf.reduce_mean(tf.nn.l2_loss(logits-labels))

   tf.summary.scalar('loss', loss)
   summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR+'/'+'logs/', graph=tf.get_default_graph())
   merged_summary_op = tf.summary.merge_all()
   
   train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)
   
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
   train_images, train_annots, train_ids, test_images, test_annots, test_ids = loadData(DATA_DIR, DATA_TYPE, classes)
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
      batch_images = np.empty((BATCH_SIZE, 299, 299, 3), dtype=np.float32)

      i = 0
      for p in batch_paths:
         img = misc.imread(p).astype('float32')
         img = misc.imresize(img, (299,299))
         img = data_ops.normalize(img)
         batch_images[i, ...] = img
         i += 1

      _, loss_, summary = sess.run([train_op, loss, merged_summary_op], feed_dict={images:batch_images, labels:batch_y})
      
      summary_writer.add_summary(summary, step)

      print 'epoch:',epoch_num,'step:',step,'loss:',loss_

      step += 1

      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')

         idx          = np.random.choice(np.arange(test_len), BATCH_SIZE, replace=False)
         batch_y      = test_annots[idx]
         batch_images = test_images[idx]

         preds = np.asarray(sess.run([logits], feed_dict={images:batch_images, labels:batch_y}))[0]

         f = open(CHECKPOINT_DIR+'testing.txt', 'a')
         batch_err = 0
         for r,p in zip(batch_y, preds):
            batch_err = batch_err + np.linalg.norm(r-p)
         batch_err = float(batch_err)/float(BATCH_SIZE)
         f.write(str(step)+','+str(batch_err)+'\n')
         f.close()

   print 'Training done'
