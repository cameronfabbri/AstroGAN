'''
   Generates x amount of galaxies
'''
import tensorflow.contrib.layers as tcl
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import scipy.misc as misc
import cPickle as pickle
import tensorflow as tf
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
from config import classes
from tf_ops import *
from nets import *
import data_ops

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--CHECKPOINT_DIR', required=True,help='checkpoint directory',type=str)
   parser.add_argument('--OUTPUT_DIR',     required=False,help='Directory to save data', type=str,default='./')
   parser.add_argument('--NUM',        required=False,help='Maximum images to generate',  type=int,default=5)
   parser.add_argument('--DATA_DIR',       required=True,help='Data directory',type=str)
   a = parser.parse_args()

   CHECKPOINT_DIR = a.CHECKPOINT_DIR
   OUTPUT_DIR     = a.OUTPUT_DIR
   NUM        = a.NUM
   DATA_DIR       = a.DATA_DIR

   BATCH_SIZE = 1

   try: os.makedirs(OUTPUT_DIR)
   except: pass

   SIZE = 64

   # placeholders for data going into the network
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')
   y           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 37), name='y')

   # generated images
   gen_images = netG(z, y, BATCH_SIZE, SIZE)
   
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
   train_images, train_annots, train_ids, test_images, test_annots, test_ids = data_ops.load_zoo(DATA_DIR, SIZE)
   test_len = len(test_annots)

   print 'generating',str(NUM*len(test_annots)),'images'
   i = 0
   for a in test_annots:
      num_gen = 0
      while num_gen < NUM:
         batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         idx     = np.random.choice(np.arange(test_len), BATCH_SIZE, replace=False)
         batch_y = test_annots[idx]

         img = np.squeeze(np.asarray(sess.run([gen_images], feed_dict={z:batch_z, y:batch_y})[0]))
         img = (img+1.)
         img *= 127.5
         img = np.clip(img, 0, 255).astype(np.uint8)
         img = np.reshape(img, (64, 64, -1))

         # save out image, z, and y*classes
         misc.imsave(OUTPUT_DIR+str(i)+'_'+str(num_gen)+'.png', img)

         num_gen += 1
      i += 1
