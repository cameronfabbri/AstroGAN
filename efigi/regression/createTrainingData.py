'''

   This file creates new data based on the training attributes. Can create an arbitrary amount of images
   for each attribute.

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


if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--CHECKPOINT_DIR', required=True,help='checkpoint directory',type=str)
   parser.add_argument('--OUTPUT_DIR',     required=False,help='Directory to save data', type=str,default='./')
   parser.add_argument('--REDSHIFT',       required=True,help='Redshift or not', type=int,default=0)
   parser.add_argument('--DATA_DIR',       required=True,help='Data directory',type=str)
   parser.add_argument('--MAX_GEN',        required=False,help='Maximum images to generate PER ATTR',type=int,default=5)
   a = parser.parse_args()

   CHECKPOINT_DIR = a.CHECKPOINT_DIR
   OUTPUT_DIR     = a.OUTPUT_DIR
   REDSHIFT       = bool(a.REDSHIFT)
   DATA_DIR       = a.DATA_DIR
   MAX_GEN        = a.MAX_GEN

   BATCH_SIZE = 1
   
   print 'Loading data...'
   train_images, train_annots, train_ids, test_images, test_annots, test_ids = data_ops.load_efigi(DATA_DIR, REDSHIFT, 64)

   y_dim = 4
   if REDSHIFT: y_dim = 5

   try: os.makedirs(OUTPUT_DIR)
   except: pass

   # placeholders for data going into the network
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')
   y           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, y_dim), name='y')

   # generated images
   gen_images = netG(z, y, BATCH_SIZE, 64)
   
   saver = tf.train.Saver(max_to_keep=1)
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess  = tf.Session()
   sess.run(init)
   
   ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         raise
         exit()

   train_len = len(train_annots)
   print train_len,'training images'
   print 'creating',str(train_len*MAX_GEN),'images for regression training'

   # create pickle file containing image names and attributes
   pkl = open(OUTPUT_DIR+'data.pkl', 'wb')
   data_info = {}

   for t_annot, t_gid in tqdm(zip(train_annots, train_ids)):
      for count in range(MAX_GEN):
         batch_z = np.random.normal(0.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         batch_y = np.expand_dims(t_annot, 0)
         img = np.asarray(sess.run([gen_images], feed_dict={z:batch_z, y:batch_y})[0])[0]
         img = (img+1.)
         img *= 127.5
         img = np.clip(img, 0, 255).astype(np.uint8)
         img = np.reshape(img, (64, 64, -1))
         misc.imsave(OUTPUT_DIR+t_gid+'_'+str(count)+'.png', img)
         #print batch_y
         data_info[t_gid+'_'+str(count)+'.png'] = batch_y
   data = pickle.dumps(data_info)
   pkl.write(data)
   pkl.close()
   exit()
