'''

   This interpolates between two z values. Attributes (y value)
   stays the same, it is the z value that is interpolated.

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
from scipy.stats import norm

sys.path.insert(0, '../ops/')
sys.path.insert(0, '../')

from tf_ops import *
import data_ops
from nets import *

'''
   Spherical interpolation. val has a range of 0 to 1.
   https://github.com/dribnet/plat/blob/master/plat/interpolate.py
'''
def slerp(val, low, high):
   if val <= 0.0:
       return low
   elif val >= 1.0:
       return high
   elif np.allclose(low, high):
       return low
   omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
   so = np.sin(omega)
   return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

'''
   Linear interpolation
'''
def lerp(val, low, high):
   return low + (high - low) * val

if __name__ == '__main__':

   if len(sys.argv) < 3:
      print 'Usage: python generate_test_galaxies.py [pkl file] [output directory] [max gen]'
      exit()
   
   pkl_file = open(sys.argv[1], 'rb')
   a = pickle.load(pkl_file)
  
   OUTPUT_DIR = sys.argv[2]
   MAX_GEN    = int(sys.argv[3])

   CHECKPOINT_DIR = a['CHECKPOINT_DIR']
   DATA_DIR       = a['DATA_DIR']
   CLASSES        = a['CLASSES']
   UPSAMPLE       = a['UPSAMPLE']
   SIZE           = a['SIZE']
   CROP           = a['CROP']
   BATCH_SIZE     = MAX_GEN
   NUM = MAX_GEN

   try: os.makedirs(OUTPUT_DIR)
   except: pass

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')
   y           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 37), name='y')

   # generated images
   gen_images = netG(z, y, UPSAMPLE)
   
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
   train_images, train_annots, train_ids, test_images, test_annots, test_ids = data_ops.load_zoo(DATA_DIR, 64)

   test_len = len(test_ids)

   # I'll need to pick some random attributes, but the same z, then interpolate between z

   print 'generating data...'
   idx     = np.random.choice(np.arange(test_len), BATCH_SIZE, replace=False)
   batch_y = test_annots[idx]
   batch_y[:NUM+1] = batch_y[1] # gotta make sure they have the same attributes
   print batch_y[0]

   # the two z vectors to interpolate between
   two_z = np.random.normal(-1.0, 1.0, size=[2, 100]).astype(np.float32)

   alpha = np.linspace(0,1, num=NUM)
   latent_vectors = []
   lin_vectors    = []
   x1 = two_z[0]
   x2 = two_z[1]

   # interpolate between x1 and x2 using both linear interpolation and also along the great circle
   for a in alpha:
      vector = slerp(a, x1, x2)
      latent_vectors.append(vector)

      lin_vector = lerp(a, x1, x2)
      lin_vectors.append(lin_vector)

      '''
      lin_vector = x1*(1-a) + x2*a
      lin_vectors.append(lin_vector)
      '''

   latent_vectors = np.asarray(latent_vectors)
   lin_vectors    = np.asarray(lin_vectors)

   gen_imgs = sess.run([gen_images], feed_dict={z:latent_vectors, y:batch_y})[0]
   canvas   = 255*np.ones((75, 64*(NUM+2), 3), dtype=np.uint8)
   start_x  = 6
   start_y  = 6
   end_y    = start_y+64
   for img in gen_imgs:
      img = (img+1.)/2. # these two lines properly scale from [-1, 1] to [0, 255]
      img *= 255.0/img.max()
      end_x = start_x+64
      canvas[start_y:end_y, start_x:end_x, :] = img
      start_x += 64+10
   misc.imsave(OUTPUT_DIR+'gaus_interpolate.png', canvas)
   
   lin_gen_imgs = sess.run([gen_images], feed_dict={z:lin_vectors, y:batch_y})[0]
   canvas   = 255*np.ones((75, 64*(NUM+2), 3), dtype=np.uint8)
   start_x  = 6
   start_y  = 6
   end_y    = start_y+64
   for img in lin_gen_imgs:
      img = (img+1.)/2. # these two lines properly scale from [-1, 1] to [0, 255]
      img *= 255.0/img.max()
      end_x = start_x+64
      canvas[start_y:end_y, start_x:end_x, :] = img
      start_x += 64+10
   misc.imsave(OUTPUT_DIR+'lin_interpolate.png', canvas)
