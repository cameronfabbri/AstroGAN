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

   # params
   parser = argparse.ArgumentParser()
   parser.add_argument('--EPOCHS',     required=False,help='Maximum number of epochs', type=int,  default=100)
   parser.add_argument('--DATA_DIR',   required=True, help='Directory where data is',  type=str,  default='./')
   parser.add_argument('--BATCH_SIZE', required=False,help='Batch size',               type=int,  default=64)
   a = parser.parse_args()

   EPOCHS         = a.EPOCHS
   DATA_DIR       = a.DATA_DIR
   BATCH_SIZE     = a.BATCH_SIZE

   CHECKPOINT_DIR = 'checkpoints/'
   IMAGES_DIR     = CHECKPOINT_DIR+'images/'

   # store all this information in a pickle file
   info_dict = {}
   info_dict['BATCH_SIZE'] = BATCH_SIZE
   info_dict['DATA_DIR'] = DATA_DIR
   info_dict['CHECKPOINT_DIR'] = CHECKPOINT_DIR

   try: os.makedirs(IMAGES_DIR)
   except: pass

   exp_pkl = open(CHECKPOINT_DIR+'info.pkl', 'wb')
   data = pickle.dumps(info_dict)
   exp_pkl.write(data)
   exp_pkl.close()

   global_step = tf.Variable(0, name='global_step', trainable=False)
   # placeholders for data going into the network
   real_images  = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 3), name='real_images')
   small_images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 64, 64, 3), name='real_images')

   # generated images - output is 256x256x3
   gen_images = netG(small_images)

   # get the output from D on the real and fake data
   errD_real = netD(real_images)
   errD_fake = netD(gen_images, reuse=True)

   # Important! no initial activations done on the last layer for D, so if one method needs an activation, do it here
   e = 1e-12
   # cost functions
   errD = tf.reduce_mean(errD_real) - tf.reduce_mean(errD_fake)
   errG = tf.reduce_mean(errD_fake)

   # also add in L1 loss to errG
   errG = errG + 100.0*tf.reduce_mean(tf.abs(real_images-gen_images))

   # gradient penalty
   epsilon = tf.random_uniform([], 0.0, 1.0)
   x_hat = real_images*epsilon + (1-epsilon)*gen_images
   d_hat = netD(x_hat, reuse=True)
   gradients = tf.gradients(d_hat, x_hat)[0]
   slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
   gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
   errD += gradient_penalty
   
   # training details
   n_critic = 5
   beta1    = 0.0
   beta2    = 0.9
   lr       = 1e-4

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   merged_summary_op = tf.summary.merge_all()

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

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
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   ########################################### training portion

   step = sess.run(global_step)
   
   print 'Loading data...'
   train_paths = np.asarray(sorted(glob.glob(DATA_DIR+'images_training_rev1/train/*.jpg')))
   test_paths  = np.asarray(sorted(glob.glob(DATA_DIR+'images_training_rev1/test/*.jpg')))

   train_len = len(train_paths)
   test_len  = len(test_paths)

   print 'train num:',train_len
   print 'test num:', test_len
   
   epoch_num = step/(train_len/BATCH_SIZE)
   
   while epoch_num < EPOCHS:
      epoch_num = step/(train_len/BATCH_SIZE)
      start = time.time()

      idx         = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
      batch_paths = train_paths[idx]

      batch_a_images = np.empty((BATCH_SIZE, 256, 256, 3), dtype=np.float32)
      batch_b_images = np.empty((BATCH_SIZE, 64, 64, 3), dtype=np.float32)
      i = 0
      for p in batch_paths:
         img = misc.imread(p).astype('float32')
         img_a = misc.imresize(img, (256,256))
         img_b = misc.imresize(img, (64,64))
         img_a = data_ops.normalize(img_a)
         img_b = data_ops.normalize(img_b)
         batch_a_images[i, ...] = img_a
         batch_b_images[i, ...] = img_b
         i += 1
      r = random.random()
      if r < 0.5:
         batch_a_images = np.fliplr(batch_a_images)
         batch_b_images = np.fliplr(batch_b_images)
      r = random.random()
      if r < 0.5:
         batch_a_images = np.flipud(batch_a_images)
         batch_b_images = np.flipud(batch_b_images)
      
      # update D
      for critic_itr in range(n_critic):
         sess.run(D_train_op, feed_dict={real_images:batch_a_images, small_images:batch_b_images})
         
      sess.run(G_train_op, feed_dict={real_images:batch_a_images, small_images:batch_b_images})
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op],
                              feed_dict={real_images:batch_a_images, small_images:batch_b_images})

      summary_writer.add_summary(summary, step)

      print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',time.time()-start
      step += 1

      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')

         idx          = np.random.choice(np.arange(test_len), BATCH_SIZE, replace=False)
         batch_paths  = test_paths[idx]
         batch_images = np.empty((BATCH_SIZE, 64, 64, 3), dtype=np.float32)
         r_images = np.empty((BATCH_SIZE, 256, 256, 3), dtype=np.float32)
         i = 0
         for p in batch_paths:
            img = misc.imread(p).astype('float32')
            img = misc.imresize(img, (64,64))
            img = data_ops.normalize(img)
            batch_images[i, ...] = img
            r_images[i, ...] = misc.imresize(misc.imread(p), (256,256))
            i += 1

         gen_imgs = np.squeeze(np.asarray(sess.run([gen_images], feed_dict={small_images:batch_images})))

         num = 0
         for r_img,g_img,s_img in zip(r_images, gen_imgs, batch_images):
            s_img = (s_img+1.)
            s_img *= 127.5
            s_img = np.clip(s_img, 0, 255).astype(np.uint8)

            g_img = (g_img+1.)
            g_img *= 127.5
            g_img = np.clip(g_img, 0, 255).astype(np.uint8)
            #r_img = np.reshape(img, (SIZE, SIZE, -1))
            misc.imsave(IMAGES_DIR+'step_'+str(step)+'_real.png', r_img)
            misc.imsave(IMAGES_DIR+'step_'+str(step)+'_gen.png', g_img)
            misc.imsave(IMAGES_DIR+'step_'+str(step)+'_orig.png', g_img)
            num += 1
            if num == 5: break
   saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
   saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')


