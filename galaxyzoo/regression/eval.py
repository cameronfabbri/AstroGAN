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

'''
   Loads the data specified, either generated or real, and also with/without redshift
'''
def loadData(data_dir, data_type, classes):
   
   # data must be of size SIZE
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

      train_paths = np.asarray(train_paths)
      train_attributes = np.asarray(train_attributes)
      train_ids = np.asarray(train_ids)
      test_paths = np.asarray(test_paths)
      test_attributes = np.asarray(test_attributes)
      test_ids = np.asarray(test_ids)

      return train_paths, train_attributes, train_ids, test_paths, test_attributes, test_ids

   elif data_type == 'gen':
      print 'using gen data'

      pkl_file = open(data_dir+'data.pkl', 'rb')
      data_info = pickle.load(pkl_file)

      train_paths = sorted(glob.glob(data_dir+'*.png'))
      train_ids   = [ntpath.basename(x.split('.')[0]) for x in train_paths]
      
      test_paths = sorted(glob.glob('/mnt/data1/images/galaxyzoo/images_training_rev1/test/*.jpg'))
      test_ids   = [ntpath.basename(x.split('.')[0]) for x in test_paths]
      
      train_attributes = []
      test_attributes  = []

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

            # remember train_ids is all str
            if str(im_id) in test_ids:
               test_attributes.append(att)
      
      train_paths = np.asarray(train_paths)
      train_attributes = np.asarray(train_attributes)
      train_ids = np.asarray(train_ids)
      test_paths = np.asarray(test_paths)
      test_attributes = np.asarray(test_attributes)
      test_ids = np.asarray(test_ids)

      return train_paths, train_attributes, train_ids, test_paths, test_attributes, test_ids

   else:
      print 'data type must be \'real\' or \'gen\''
      exit()

if __name__ == '__main__':

   SIZE = 224
   parser = argparse.ArgumentParser()
   parser.add_argument('--DATA_TYPE',  required=True,help='Real or generated data',type=str)
   parser.add_argument('--DATA_DIR',   required=True,help='Data directory',type=str)
   a = parser.parse_args()

   DATA_TYPE      = a.DATA_TYPE
   DATA_DIR       = a.DATA_DIR

   CHECKPOINT_DIR = 'checkpoints/'+'DATA_TYPE_'+DATA_TYPE+'/'

   images = tf.placeholder(tf.float32, shape=(1, SIZE, SIZE, 3), name='real_images')
   labels = tf.placeholder(tf.float32, shape=(1, 37), name='attributes')

   #with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
   #   logits, _ = inception_resnet_v2.inception_resnet_v2(images, num_classes=37, is_training=False)
   with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
      logits, _ = alexnet.alexnet_v2(images, num_classes=37, is_training=True)
      # clip logits between [0, 1] because that's the range of the labels
      logits = tf.minimum(tf.maximum(0.0,logits), 1.0)

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

   train_paths, train_annots, train_ids, test_paths, test_annots, test_ids = loadData(DATA_DIR, DATA_TYPE, classes)

   train_len = len(train_annots)
   test_len  = len(test_annots)

   f = open(CHECKPOINT_DIR+'evaluation.txt', 'a')
   all_pred = []
   for test_p, test_attr in tqdm(zip(test_paths, test_annots)):
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
