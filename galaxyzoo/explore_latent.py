'''

   Creates a cool gif interpolating along the latent space

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
import imageio

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

   if len(sys.argv) < 2:
      print 'Usage: python generate_test_galaxies.py [pkl file] [max_gen]'
      exit()

   pkl_file = open(sys.argv[1], 'rb')
   a = pickle.load(pkl_file)
 
   MAX_GEN = int(sys.argv[2])

   CHECKPOINT_DIR = a['CHECKPOINT_DIR']
   DATA_DIR       = a['DATA_DIR']
   CLASSES        = a['CLASSES']
   UPSAMPLE       = a['UPSAMPLE']
   SIZE           = a['SIZE']
   CROP           = a['CROP']
   HOT            = a['HOT']
   BATCH_SIZE = 1

   # play around with this to see what looks smooth
   NUM=20

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
   train_images, train_annots, train_ids, test_images, test_annots, test_ids = data_ops.load_zoo(DATA_DIR, HOT)
   test_len = len(test_ids)
   print 'Done\n'

   '''
      Approach:

      Going to pick two z vectors and interpolate between them, then set the second vector as the 'start'
      and pick a random second vector for the end, and repeat. For each I'll use the same random attribute
   '''

   idx     = np.random.choice(np.arange(test_len), 2, replace=False)
   batch_y = test_annots[idx]
   #batch_y[:NUM+1] = batch_y[1] # gotta make sure they have the same attributes

   # the two z vectors to interpolate between
   two_z = np.random.normal(-1.0, 1.0, size=[2, 100]).astype(np.float32)

   alpha = np.linspace(0,1, num=NUM)
   latent_vectors = []
   latent_attributes = []
   x1 = two_z[0]
   x2 = two_z[1]
   y1 = batch_y[0]
   y2 = batch_y[1]

   # interpolate between x1 and x2 using both linear interpolation and also along the great circle
   # also interpolate along the attributes
   for a in alpha:
      vector = slerp(a, x1, x2)
      vector_a = slerp(a, y1, y2)
      latent_vectors.append(vector)
      latent_attributes.append(vector_a)

   while len(latent_vectors) < MAX_GEN:

      # set the ending point as the new starting point
      x1 = x2
      y1 = y2

      # get a new ending point
      x2  = np.random.normal(-1.0, 1.0, size=[1,100]).astype(np.float32)
      idx = np.random.choice(np.arange(test_len), 1, replace=False)
      y2  = test_annots[idx]

      for a in alpha:
         vector = slerp(a, np.squeeze(x1), np.squeeze(x2))
         vector_a = slerp(a, np.squeeze(y1), np.squeeze(y2))
         latent_vectors.append(vector)
         latent_attributes.append(vector_a)

   latent_vectors = np.asarray(latent_vectors)
   latent_attributes = np.asarray(latent_attributes)

   gif_images = []
   ccc = 0
   print 'Generating data...'
   for l,a in tqdm(zip(latent_vectors, latent_attributes)):
      l = np.expand_dims(l, 0)
      a = np.expand_dims(a, 0)
      gen_img = np.squeeze(sess.run([gen_images], feed_dict={z:l, y:a})[0])
      gen_img = (gen_img+1.)/2.
      gen_img *= 255.0/gen_img.max()
      gen_img = misc.imresize(gen_img, (128,128))
      gif_images.append(gen_img)
      ccc += 1
   print '\nSaving out gif...\n')
   imageio.mimsave('interpolation.gif', gif_images)
