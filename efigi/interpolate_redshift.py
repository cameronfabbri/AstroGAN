'''

   This interpolates along the redshift attribute only to see if there is any change.

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
   parser.add_argument('--DATA_DIR',       required=True,help='Directory with data', type=str,default='./')
   parser.add_argument('--NUM',            required=False,help='Maximum images to interpolate',  type=int,default=9)
   a = parser.parse_args()

   CHECKPOINT_DIR = a.CHECKPOINT_DIR
   OUTPUT_DIR     = a.OUTPUT_DIR
   DATA_DIR       = a.DATA_DIR
   NUM            = a.NUM
   BATCH_SIZE     = NUM

   try: os.makedirs(OUTPUT_DIR)
   except: pass

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')
   y           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 5), name='y')

   # generated images
   gen_images = netG(z, y, BATCH_SIZE, 64)
   
   saver = tf.train.Saver(max_to_keep=1)
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess  = tf.Session()
   sess.run(init)
   
   # restore previous model if there is one
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
   
   print 'Loading data...'
   REDSHIFT = 1
   train_images, train_annots, train_ids, test_images, test_annots, test_ids = data_ops.load_efigi(DATA_DIR, REDSHIFT, 64)
   test_len = len(test_ids)

   print 'generating data...'
   idx     = np.random.choice(np.arange(test_len), BATCH_SIZE, replace=False)
   batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
   batch_z[1:] = batch_z[0]
   batch_y = test_annots[idx]
   batch_y[:NUM+1] = batch_y[1] # gotta make sure they have the same attributes

   batch_y[0][-1] = 0
   batch_y[1][-1] = 0.5
   print batch_y[0]
   print batch_y[1]

   alpha = np.linspace(0,1, num=NUM)
   latent_vectors = []
   x1 = batch_y[0]
   x2 = batch_y[1]

   for a in alpha:
      vector = x1*(1-a) + x2*a
      latent_vectors.append(vector)

   latent_vectors = np.asarray(latent_vectors)

   gen_imgs = sess.run([gen_images], feed_dict={z:batch_z, y:latent_vectors})[0]
   canvas   = 255*np.ones((80, 64*(NUM+2), 3), dtype=np.uint8)
   start_x  = 10
   start_y  = 10
   end_y    = start_y+64

   for img in gen_imgs:
      img = (img+1.)/2. # these two lines properly scale from [-1, 1] to [0, 255]
      img *= 255.0/img.max()
      end_x = start_x+64
      canvas[start_y:end_y, start_x:end_x, :] = img
      start_x += 64+10
   misc.imsave(OUTPUT_DIR+'interpolate.png', canvas)
