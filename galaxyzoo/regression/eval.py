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

'''
   Loads the data specified, either generated or real, and also with/without redshift
'''
def loadData(data_dir, data_type, classes):

   # data must be of size 299
   if data_type == 'real':
      train_images, train_annots, train_ids, test_images, test_annots, test_ids = data_ops.load_zoo(DATA_DIR, 64)
      return train_images, train_annots, train_ids, test_images, test_annots, test_ids
   elif data_type == 'gen':
      print 'using gen data'


if __name__ == '__main__':
         
   parser = argparse.ArgumentParser()
   parser.add_argument('--DATA_TYPE',  required=True,help='Real or generated data',type=str)
   parser.add_argument('--DATA_DIR',   required=True,help='Data directory',type=str)
   a = parser.parse_args()

   BATCH_SIZE     = a.BATCH_SIZE
   DATA_TYPE      = a.DATA_TYPE
   DATA_DIR       = a.DATA_DIR
   EPOCHS         = a.EPOCHS

   CHECKPOINT_DIR = 'checkpoints/'+'DATA_TYPE_'+DATA_TYPE+'/'

   #images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 299, 299, 3), name='real_images')
   images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 64, 64, 3), name='real_images')
   labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 18), name='attributes')

   #with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
   #   logits, _ = inception_resnet_v2.inception_resnet_v2(images, num_classes=37, is_training=False)
   with slim.arg_scope(cifarnet.cifarnet_arg_scope()):
      logits, _ = cifarnet.cifarnet(images, num_classes=37, is_training=False)

   saver = tf.train.Saver()
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

   train_images, train_annots, train_ids, test_images, test_annots, test_ids = loadData(DATA_DIR, DATA_TYPE, classes)

   train_len = len(train_annots)
   test_len  = len(test_annots)
   step = sess.run(global_step)

   for test_img, test_attr in zip(test_images, test_annots):

      test_img  = np.expand_dims(test_img, 0)
      test_attr = np.expand_dims(test_attr, 0)
      preds     = np.asarray(sess.run([logits], feed_dict={images:test_img, labels:test_attr}))[0]

      f = open(CHECKPOINT_DIR+'evaluation.txt', 'a')
      batch_err = 0
      for r,p in zip(batch_y, preds):
         batch_err = batch_err + np.linalg.norm(r-p)
      batch_err = float(batch_err)/float(BATCH_SIZE)
      f.write(str(step)+','+str(batch_err)+'\n')
      f.close()
