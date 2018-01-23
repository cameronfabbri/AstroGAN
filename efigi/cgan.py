'''

   Conditional GAN for the EFIGI dataset.

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
import time
import sys
import os

sys.path.insert(0, '../ops/')
sys.path.insert(0, '../')

from tf_ops import *
from nets import *
import data_ops

from config import classes

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--BATCH_SIZE', required=False,help='Batch size',              type=int,default=64)
   parser.add_argument('--DATA_DIR',   required=True, help='Directory where data is', type=str,default='./')
   parser.add_argument('--EPOCHS',     required=False,help='Maximum training epochs', type=int,default=500)
   parser.add_argument('--LOSS',       required=False,help='Type of GAN loss to use', type=str,default='wgan')
   a = parser.parse_args()

   BATCH_SIZE     = a.BATCH_SIZE
   DATA_DIR       = a.DATA_DIR
   EPOCHS         = a.EPOCHS
   LOSS           = a.LOSS

   # convert to string for directory naming
   cn = ''
   for i in classes:
      cn = cn + str(i)

   CHECKPOINT_DIR = 'checkpoints/LOSS_'+LOSS+'/C_'+str(cn)+'/'
   IMAGES_DIR     = CHECKPOINT_DIR+'images/'

   try: os.makedirs(IMAGES_DIR)
   except: pass
   
   exp_info = dict()
   exp_info['CHECKPOINT_DIR']   = CHECKPOINT_DIR
   exp_info['BATCH_SIZE'] = BATCH_SIZE
   exp_info['DATA_DIR']   = DATA_DIR
   exp_info['CLASSES']    = classes
   exp_info['EPOCHS']     = EPOCHS
   exp_info['LOSS']       = LOSS

   exp_pkl = open(CHECKPOINT_DIR+'info.pkl', 'wb')
   data = pickle.dumps(exp_info)
   exp_pkl.write(data)
   exp_pkl.close()

   print 'Loading data...'
   train_images, train_annots, train_ids, test_images, test_annots, test_ids = data_ops.load_efigi(DATA_DIR, classes, 64)

   # find y dimension
   idx_ = np.array([1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49])
   idx_ = np.multiply(classes[:-1], idx_)
   idx = [x for x in idx_ if x != 0]
   y_dim = len(idx)
   # account for redshift attribute
   if classes[-1] == 1: y_dim += 1

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   real_images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 64, 64, 3), name='real_images')
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')
   y           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, y_dim), name='y')

   # generated images
   gen_images = netG(z, y, BATCH_SIZE, 64)

   # get the output from D on the real and fake data
   errD_real = netD(real_images, y, BATCH_SIZE, LOSS, 64)
   errD_fake = netD(gen_images, y, BATCH_SIZE, LOSS, 64, reuse=True)

   # Important! no initial activations done on the last layer for D, so if one method needs an activation, do it
   e = 1e-12
   if LOSS == 'gan':
      errD_real = tf.nn.sigmoid(errD_real)
      errD_fake = tf.nn.sigmoid(errD_fake)
      errG = tf.reduce_mean(-tf.log(errD_fake + e))
      errD = tf.reduce_mean(-(tf.log(errD_real+e)+tf.log(1-errD_fake+e)))

   if LOSS == 'lsgan':
      errD_real = tf.nn.sigmoid(errD_real)
      errD_fake = tf.nn.sigmoid(errD_fake)
      errD = tf.reduce_mean(0.5*(tf.square(errD_real - 1)) + 0.5*(tf.square(errD_fake)))
      errG = tf.reduce_mean(0.5*(tf.square(errD_fake - 1)))

   if LOSS == 'wgan':
      # cost functions
      errD = tf.reduce_mean(errD_real) - tf.reduce_mean(errD_fake)
      errG = tf.reduce_mean(errD_fake)

      # gradient penalty
      epsilon = tf.random_uniform([], 0.0, 1.0)
      x_hat = real_images*epsilon + (1-epsilon)*gen_images
      d_hat = netD(x_hat, y, BATCH_SIZE, LOSS, 64, reuse=True)
      gradients = tf.gradients(d_hat, x_hat)[0]
      slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
      gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
      errD += gradient_penalty

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   merged_summary_op = tf.summary.merge_all()

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   if LOSS == 'wgan':
      n_critic = 5
      beta1    = 0.0
      beta2    = 0.9
      lr       = 1e-4

   if LOSS == 'lsgan':
      n_critic = 1
      beta1    = 0.5
      beta2    = 0.999
      lr       = 0.001

   if LOSS == 'gan':
      n_critic = 1
      beta1    = 0.5
      beta2    = 0.999
      lr       = 0.0002

   # optimize G
   G_train_op = tf.train.AdamOptimizer(learning_rate=lr,beta1=beta1,beta2=beta2).minimize(errG, var_list=g_vars, global_step=global_step)
   # optimize D
   D_train_op = tf.train.AdamOptimizer(learning_rate=lr,beta1=beta1,beta2=beta2).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=1)
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess  = tf.Session()
   sess.run(init)

   # write losses to tf summary to view in tensorboard
   try: tf.summary.scalar('d_loss', tf.reduce_mean(errD))
   except:pass
   try: tf.summary.scalar('g_loss', tf.reduce_mean(errG))
   except:pass

   summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR+'/'+'logs/', graph=tf.get_default_graph())

   tf.add_to_collection('G_train_op', G_train_op)
   tf.add_to_collection('D_train_op', D_train_op)
   
   # restore previous model if there is one
   ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print 'Restoring previous model...'
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print 'Model restored'
      except:
         print 'Could not restore model'
         pass
   
   ########################################### training portion

   step = sess.run(global_step)
   train_len = len(train_annots)
   test_len  = len(test_annots)

   print 'train num:',train_len
   print 'test num:',test_len
   
   epoch_num = step/(train_len/BATCH_SIZE)
   
   while epoch_num < EPOCHS:
      epoch_num = step/(train_len/BATCH_SIZE)
      start = time.time()

      # train the discriminator
      for critic_itr in range(n_critic):
         idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
         batch_z      = np.random.normal(0.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         batch_y      = train_annots[idx]
         batch_images = train_images[idx]

         # randomly flip batch
         #r = random.random()
         # flip image left right
         #if r < 0.5:
         #   batch_images = np.fliplr(batch_images)
         
         #r = random.random()
         # flip image up down
         #if r < 0.5:
         #   batch_images = np.flipud(batch_images)

         sess.run(D_train_op, feed_dict={z:batch_z, y:batch_y, real_images:batch_images})
      
      # now train the generator once! use normal distribution, not uniform!!
      idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
      batch_z      = np.random.normal(0.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
      batch_y      = train_annots[idx]
      batch_images = train_images[idx]

      sess.run(G_train_op, feed_dict={z:batch_z, y:batch_y, real_images:batch_images})
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op],
                                 feed_dict={z:batch_z, y:batch_y, real_images:batch_images})

      summary_writer.add_summary(summary, step)

      print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',time.time()-start
      step += 1
    
      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')

         idx          = np.random.choice(np.arange(test_len), BATCH_SIZE, replace=False)
         batch_z      = np.random.normal(0.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         batch_y      = test_annots[idx]
         batch_images = test_images[idx]

         gen_imgs = np.squeeze(np.asarray(sess.run([gen_images],
                                 feed_dict={z:batch_z, y:batch_y, real_images:batch_images})))
         num = 0
         for img,atr in zip(gen_imgs, batch_y):
            img = (img+1.)
            img *= 127.5
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = np.reshape(img, (64, 64, -1))
            misc.imsave(IMAGES_DIR+'step_'+str(step)+'_num_'+str(num)+'.png', img)
            with open(IMAGES_DIR+'attrs.txt', 'a') as f:
               f.write('step_'+str(step)+'_num_'+str(num)+','+str(atr)+'\n')
            num += 1
            if num == 5: break
   
   saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
   saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')
