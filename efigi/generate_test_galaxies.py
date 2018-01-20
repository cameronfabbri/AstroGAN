'''

   This file takes the test set attributes and generates 5 images from those with random z vectors
   Each image saved out is the original test image on the left, then the next 5 columns are the generated

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

sys.path.insert(0, '../ops/')
sys.path.insert(0, '../')

from tf_ops import *
from nets import *
import data_ops


if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--CHECKPOINT_DIR', required=True,help='checkpoint directory',type=str)
   parser.add_argument('--OUTPUT_DIR',     required=False,help='Directory to save data', type=str,default='./')
   parser.add_argument('--REDSHIFT',       required=True,help='Redshift or not', type=int,default=0)
   parser.add_argument('--DATA_DIR',       required=True,help='Data directory',type=str)
   parser.add_argument('--MAX_GEN',        required=False,help='Maximum images to generate',  type=int,default=5)
   a = parser.parse_args()

   CHECKPOINT_DIR = a.CHECKPOINT_DIR
   OUTPUT_DIR     = a.OUTPUT_DIR
   REDSHIFT       = bool(a.REDSHIFT)
   DATA_DIR       = a.DATA_DIR
   MAX_GEN        = a.MAX_GEN

   BATCH_SIZE = 64
  
   
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

   test_len = len(test_annots)

   print 'generating data...'
   batch_z = np.random.normal(0.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
   idx     = np.random.choice(np.arange(test_len), BATCH_SIZE, replace=False)
   batch_y = test_annots[idx]

   gen_imgs = np.asarray(sess.run([gen_images], feed_dict={z:batch_z, y:batch_y})[0])
   
   # height, width, depth
   canvas = 255*np.ones((84, MAX_GEN*74+10 , 3), dtype=np.uint8)
   
   start_x = 10
   start_y = 10
   end_y = start_y+64

   counter = 0
   for img in gen_imgs:
      img = (img+1.)
      img *= 127.5
      img = np.clip(img, 0, 255).astype(np.uint8)
      img = np.reshape(img, (64, 64, -1))
      end_x = start_x+64
      canvas[start_y:end_y, start_x:end_x, :] = img
      counter += 1
      start_x = end_x + 10
      if counter == MAX_GEN: break
   misc.imsave(OUTPUT_DIR+'generated.png', canvas)
   exit()

   # batch y should be chosen from the test set
   exit()
   batch_y = np.random.choice([0, 1], size=(BATCH_SIZE,9))
   batch_y[0][:] = 0
   batch_y[0][-3] = male # make male or female
   for i in range(14):
      batch_y[0][i] = 1
      print batch_y[0]
      gen_imgs = sess.run([gen_images], feed_dict={z:batch_z, y:batch_y})[0]
      for img in gen_imgs:
         img = (img+1.)
         img *= 127.5
         img = np.clip(img, 0, 255).astype(np.uint8)
         img = np.reshape(img, (64, 64, -1))
         break
      end_x = start_x+64
      canvas[start_y:end_y, start_x:end_x, :] = img
      start_x += 64+10
   
      batch_y = np.random.choice([0, 1], size=(BATCH_SIZE,9))
      batch_y[0][:] = 0
      batch_y[0][-3] = male # make male or female

   misc.imsave(OUTPUT_DIR+'attributes.png', canvas)
   exit()
