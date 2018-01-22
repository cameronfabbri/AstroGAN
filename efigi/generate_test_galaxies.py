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

   for t_img, t_annot, t_gid in zip(test_images, test_annots, test_ids):
      #misc.imsave(OUTPUT_DIR+t_gid+'_.png',t_img)
      canvas = 255*np.ones((84, (MAX_GEN+1)*74+10 , 3), dtype=np.uint8)
      start_x = 10
      start_y = 10
      end_y = start_y+64
      t_img = (t_img+1.)
      t_img *= 127.5
      t_img = np.clip(t_img, 0, 255).astype(np.uint8)
      t_img = np.reshape(t_img, (64, 64, -1))
      end_x = start_x+64
      canvas[start_y:end_y, start_x:end_x, :] = t_img
      start_x = end_x+10

      # put a line of black pixels in between the real image and generated ones
      canvas[:, end_x+5] = 0
      
      for count in range(MAX_GEN):
         batch_z = np.random.normal(0.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         batch_y = np.expand_dims(t_annot, 0)
         img = np.asarray(sess.run([gen_images], feed_dict={z:batch_z, y:batch_y})[0])[0]
         img = (img+1.)
         img *= 127.5
         img = np.clip(img, 0, 255).astype(np.uint8)
         img = np.reshape(img, (64, 64, -1))
         
         end_x = start_x+64
         canvas[start_y:end_y, start_x:end_x, :] = img
         start_x = end_x + 10
         
      misc.imsave(OUTPUT_DIR+t_gid+'.png', canvas)
