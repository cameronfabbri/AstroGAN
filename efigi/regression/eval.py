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
import alexnet
sys.path.insert(0, '../../ops/')
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

from tf_ops import *
from nets import *
import data_ops
from config import classes
import inception_resnet_v2
slim = tf.contrib.slim
from train import loadData

if __name__ == '__main__':

   SIZE = 224
   
   pkl_file = open(sys.argv[1], 'rb')
   info_dict = pickle.load(pkl_file)

   DATA_TYPE = info_dict['DATA_TYPE']
   DATA_DIR  = info_dict['DATA_DIR']
   use_both  = bool(info_dict['USE_BOTH'])
   NETWORK   = info_dict['NETWORK']
   BATCH_SIZE = info_dict['BATCH_SIZE']

   print DATA_TYPE
   print NETWORK
   print BATCH_SIZE

   CHECKPOINT_DIR = 'checkpoints/'+'DATA_TYPE_'+DATA_TYPE+'/NETWORK_'+NETWORK+'/USE_BOTH_'+str(use_both)+'/'

   images = tf.placeholder(tf.float32, shape=(1, SIZE, SIZE, 3), name='real_images')
   labels = tf.placeholder(tf.float32, shape=(1, 5), name='attributes')
   
   # clip logits between [0, 1] because that's the range of the labels
   if NETWORK == 'inception':
      print 'Using inception'
      with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
         logits, _ = inception_resnet_v2.inception_resnet_v2(images, num_classes=5, is_training=False)
   if NETWORK == 'alexnet':
      print 'Using alexnet'
      with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
         logits, _ = alexnet.alexnet_v2(images, num_classes=5, is_training=False)
   print 'Done.'

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

   train_images, train_annots, train_ids, test_images, test_annots, test_ids = loadData(DATA_DIR, DATA_TYPE, use_both, classes)

   train_len = len(train_annots)
   test_len  = len(test_annots)

   f = open(CHECKPOINT_DIR+'evaluation.txt', 'a')
   all_pred = []
   for test_p, test_attr in tqdm(zip(test_images, test_annots)):
      test_img  = misc.imread(test_p).astype('float32')
      test_img  = misc.imresize(test_img, (SIZE,SIZE))
      test_img  = test_img/255.0
      test_img  = np.expand_dims(test_img, 0)
      test_attr = np.expand_dims(test_attr, 0)
      pred      = np.asarray(sess.run([logits], feed_dict={images:test_img}))[0]
      all_pred.append(np.squeeze(pred))
   total_err = sqrt(mean_squared_error(test_annots, np.asarray(all_pred)))
   print 'total RMSE:',total_err
   f.write('RMSE:'+str(total_err)+'\n')
   f.close()
