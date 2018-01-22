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

from tf_ops import *
from nets import *
import data_ops

import inception_resnet_v2

slim = tf.contrib.slim

'''
   Loads the data specified, either generated or real, and also with/without redshift
'''
def loadData(data_dir, data_type, redshift):

   print 'data_dir: ',data_dir
   print 'data_type:',data_type
   print 'redshift: ',redshift

   if data_type == 'real':
      print 'training on real images'
      redict = {}
      if redshift:
         d=0
         with open(data_dir+'EFIGI_coord_redshift.txt','r') as f:
            for line in f:
               if d==0:
                  d=1
                  continue
               line = line.rstrip().split()
               galaxy_id = line[0]
               redshift  = float(line[9])
               if redshift < 0: continue # redshift missing, so has a value of -99.99 we don't want
               redict[galaxy_id] = redshift

      #idx = np.array([0, 1, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 40, 49])
      idx = np.array([7, 10, 31, 49])

      train_images     = glob.glob(data_dir+'images/train/*.png')
      test_images      = glob.glob(data_dir+'images/test/*.png')

      # get train ids from train folder
      train_ids = [ntpath.basename(x.split('.')[0]) for x in train_images]
      test_ids  = [ntpath.basename(x.split('.')[0]) for x in test_images]

      iptr = data_dir+'images/train/'
      ipte = data_dir+'images/test/'

      train_images = []
      test_images  = []
      train_attributes = []
      test_attributes  = []

      r_test_ids = []
      paths = []
      with open(data_dir+'EFIGI_attributes.txt', 'r') as f:
         for line in f:
            line     = line.rstrip().split()
            galaxy_id = line[0]
            line     = np.asarray(line[1:])
            line     = line[idx].astype('float32')
            if redshift:
               try: line = np.append(line, redict[galaxy_id]) # if using redshift, add it to the attributes
               except: continue # don't use this one in training
            
            if galaxy_id in train_ids:
               img = misc.imread(iptr+galaxy_id+'.png').astype('float32')
               img = misc.imresize(img, (size, size))
               img = normalize(img)
               train_images.append(img)
               train_attributes.append(line)
            elif galaxy_id in test_ids:
               paths.append(ipte+galaxy_id+'.png')
               img = misc.imread(ipte+galaxy_id+'.png').astype('float32')
               img = misc.imresize(img, (size, size))
               img = normalize(img)
               test_images.append(img)
               test_attributes.append(line)
               r_test_ids.append(galaxy_id)

      return np.asarray(train_images), np.asarray(train_attributes), np.asarray(train_ids), np.asarray(test_images), np.asarray(test_attributes), np.asarray(r_test_ids)



if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--BATCH_SIZE', required=False,help='Batch size', type=int,default=64)
   parser.add_argument('--REDSHIFT',   required=True,help='Redshift or not', type=int,default=0)
   parser.add_argument('--DATA_DIR',   required=True,help='Data directory',type=str)
   parser.add_argument('--DATA_TYPE',  required=True,help='Real or generated data',type=str)
   a = parser.parse_args()

   BATCH_SIZE     = a.BATCH_SIZE
   REDSHIFT       = bool(a.REDSHIFT)
   DATA_DIR       = a.DATA_DIR
   DATA_TYPE      = a.DATA_TYPE

   y_dim = 4
   if REDSHIFT: y_dim = 5

   global_step = tf.Variable(0, name='global_step', trainable=False)
   images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 299, 299, 3), name='real_images')
   labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, y_dim), name='attributes')

   with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
      logits, _ = inception_resnet_v2.inception_resnet_v2(images, num_classes=5, is_training=False)

   loss = tf.nn.l2_loss(logits-labels)

   train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

   train_images, train_annots, train_ids, test_images, test_annots, test_ids = loadData(DATA_DIR, DATA_TYPE, REDSHIFT)





