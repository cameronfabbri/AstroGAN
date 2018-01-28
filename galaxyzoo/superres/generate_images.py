'''
   conditional gan
'''
import tensorflow.contrib.layers as tcl
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import scipy.misc as misc
import cPickle as pickle
import tensorflow as tf
import numpy as np
import argparse
import random
import ntpath
import glob
import time
import sys
import os

# my own imports
sys.path.insert(0, '../../ops/')
import data_ops
from nets import *
from tf_ops import *

if __name__ == '__main__':


   CHECKPOINT_DIR = sys.argv[1]
   IN_DIR  = sys.argv[2]
   OUT_DIR = sys.argv[3]

   try: os.makedirs(OUT_DIR)
   except: pass
   small_images = tf.placeholder(tf.float32, shape=(1, 64, 64, 3), name='real_images')

   # generated images - output is 256x256x3
   gen_images = netG(small_images)

   sess  = tf.Session()
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess.run(init)

   # restore previous model if there is one
   saver = tf.train.Saver()
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
         pass
   
   print 'Loading data...'
   paths = np.asarray(sorted(glob.glob(IN_DIR+'*.png')))

   i = 0
   for img_p in paths:
      print img_p
      img = misc.imread(img_p)
      img = misc.imresize(img, (64,64))
      img = data_ops.normalize(img)
      img = np.expand_dims(img, 0)

      gen_img = np.squeeze(np.asarray(sess.run([gen_images], feed_dict={small_images:img})))

      img = np.squeeze(img)
      g_img = (gen_img+1.)
      g_img *= 127.5
      g_img = np.clip(g_img, 0, 255).astype(np.uint8)
      misc.imsave(OUT_DIR+str(i)+'_real.png', img)
      misc.imsave(OUT_DIR+str(i)+'_gen.png', g_img)
      i += 1
      exit()
