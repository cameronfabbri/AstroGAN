'''

   This file is able to generate a number of galaxies all with the same attributes

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

from config import classes

if __name__ == '__main__':
   
   pkl_file = open(sys.argv[1], 'rb')
   a = pickle.load(pkl_file)
   print a
  
   OUTPUT_DIR = sys.argv[2]
   MAX_GEN    = int(sys.argv[3])

   CHECKPOINT_DIR = a['CHECKPOINT_DIR']
   DATA_DIR       = a['DATA_DIR']
   CLASSES        = a['CLASSES']
   LOSS           = a['LOSS']

   BATCH_SIZE = 1
   
   print 'Loading data...'
   train_images, train_annots, train_ids, test_images, test_annots, test_ids = data_ops.load_efigi(DATA_DIR, CLASSES, 64)

   try: os.makedirs(OUTPUT_DIR)
   except: pass

   y_dim = 18
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
   print test_len,'testing images'

   # create attributes here
   T               = 0.
   bulge_to_total  = 0.
   arm_strength    = 1.0
   arm_curve       = 0.25
   arm_rotation    = 0.
   bar_length      = 0.
   inner_ring      = 0.
   outer_ring      = 0.
   pseudo_ring     = 0.
   perturbation    = 0.
   visible_dust    = 0.
   dust_dispertion = 0.
   flocculence     = 0.
   hot_spots       = 0.
   inclination     = 0.
   contamination   = 0.
   mult            = 0.
   redshift        = 2.18

   batch_y = np.expand_dims(np.asarray([T,
                                        bulge_to_total,
                                        arm_strength,
                                        arm_curve,
                                        arm_rotation,
                                        bar_length,
                                        inner_ring,
                                        outer_ring,
                                        pseudo_ring,
                                        perturbation,
                                        visible_dust,
                                        dust_dispertion,
                                        flocculence,
                                        hot_spots,
                                        inclination,
                                        contamination,
                                        mult,
                                        redshift]), 0)

   for n in tqdm(range(MAX_GEN)):
      batch_z = np.random.normal(0.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
      img = np.asarray(sess.run([gen_images], feed_dict={z:batch_z, y:batch_y})[0])[0]
      misc.imsave(OUTPUT_DIR+str(n)+'.png',img)
