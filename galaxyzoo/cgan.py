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
import time
import sys
import os

# my own imports
sys.path.insert(0, '../ops/')
sys.path.insert(0, '../')
from config import classes
import tf_ops
from nets import *
from data_ops import *

if __name__ == '__main__':

   # params
   parser = argparse.ArgumentParser()
   parser.add_argument('--GAN',        required=False,help='Type of GAN loss to use',  type=str,  default='wgan')
   parser.add_argument('--SIZE',       required=False,help='Size of the images',       type=int,  default=64)
   parser.add_argument('--BETA1',      required=False,help='beta1 ADAM parameter',     type=float,default=0.0) # wgan only
   parser.add_argument('--EPOCHS',     required=False,help='Maximum number of epochs', type=int,  default=100)
   parser.add_argument('--DATA_DIR',   required=True, help='Directory where data is',  type=str,  default='./')
   parser.add_argument('--BATCH_SIZE', required=False,help='Batch size',               type=int,  default=64)
   a = parser.parse_args()

   GAN            = a.GAN
   SIZE           = a.SIZE
   BETA1          = a.BETA1
   EPOCHS         = a.EPOCHS
   DATA_DIR       = a.DATA_DIR
   BATCH_SIZE     = a.BATCH_SIZE

   # convert to string for directory naming
   cn = ''
   for i in classes:
      cn = cn + str(i)

   CHECKPOINT_DIR = 'checkpoints/GAN_'+GAN+'/SIZE_'+str(SIZE)+'/BETA1_'+str(BETA1)+'/CLASSES_'+str(cn)+'/'
   IMAGES_DIR     = CHECKPOINT_DIR+'images/'

   # store all this information in a pickle file
   info_dict = {}
   info_dict['GAN']        = GAN
   info_dict['SIZE']       = SIZE
   info_dict['BETA1']      = BETA1
   info_dict['BATCH_SIZE'] = BATCH_SIZE
   info_dict['classes'] = classes

   try: os.makedirs(IMAGES_DIR)
   except: pass

   exp_pkl = open(CHECKPOINT_DIR+'info.pkl', 'wb')
   data = pickle.dumps(info_dict)
   exp_pkl.write(data)
   exp_pkl.close()

   global_step = tf.Variable(0, name='global_step', trainable=False)
   # placeholders for data going into the network
   real_images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, SIZE, SIZE, 3), name='real_images')
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')
   y           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 37), name='y')
   mask        = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 37), name='mask')

   # multiply y by the mask of attributes actually being used
   y = tf.multiply(y,mask)

   # generated images
   gen_images = netG(z, y, BATCH_SIZE, SIZE)

   # get the output from D on the real and fake data
   errD_real = netD(real_images, y, BATCH_SIZE, GAN, SIZE)
   errD_fake = netD(gen_images, y, BATCH_SIZE, GAN, SIZE, reuse=True)

   # Important! no initial activations done on the last layer for D, so if one method needs an activation, do it here
   e = 1e-12
   if GAN == 'gan':
      errD_real = tf.nn.sigmoid(errD_real)
      errD_fake = tf.nn.sigmoid(errD_fake)
      errG = tf.reduce_mean(-tf.log(errD_fake + e))
      errD = tf.reduce_mean(-(tf.log(errD_real+e)+tf.log(1-errD_fake+e)))
      
      # training details
      n_critic = 1
      beta1    = 0.5
      beta2    = 0.999
      lr       = 0.0002

   if GAN == 'lsgan':
      errD_real = tf.nn.sigmoid(errD_real)
      errD_fake = tf.nn.sigmoid(errD_fake)
      errD = tf.reduce_mean(0.5*(tf.square(errD_real - 1)) + 0.5*(tf.square(errD_fake)))
      errG = tf.reduce_mean(0.5*(tf.square(errD_fake - 1)))
      
      # training details
      n_critic = 1
      beta1    = 0.5
      beta2    = 0.999
      lr       = 0.001

   if GAN == 'wgan':
      # cost functions
      errD = tf.reduce_mean(errD_real) - tf.reduce_mean(errD_fake)
      errG = tf.reduce_mean(errD_fake)

      # gradient penalty
      epsilon = tf.random_uniform([], 0.0, 1.0)
      x_hat = real_images*epsilon + (1-epsilon)*gen_images
      d_hat = netD(x_hat, y, BATCH_SIZE, GAN, SIZE, reuse=True)
      gradients = tf.gradients(d_hat, x_hat)[0]
      slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
      gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
      errD += gradient_penalty
      
      # training details
      n_critic = 5
      beta1    = BETA1
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
   if DATASET == 'zoo': train_images, train_annots, train_ids, test_images, test_annots, test_ids = data_ops.load_zoo(DATA_DIR, SIZE)
   if DATASET == 'efigi': train_images, train_annots, train_ids, test_images, test_annots, test_ids = data_ops.load_efigi(DATA_DIR, SIZE)
   print train_images.shape
   print train_annots.shape
   print test_images.shape
   print test_annots.shape
   print 'Done'

   train_len = len(train_annots)
   test_len  = len(test_annots)

   print 'train num:',train_len
   
   epoch_num = step/(train_len/BATCH_SIZE)
   
   while epoch_num < EPOCHS:
      epoch_num = step/(train_len/BATCH_SIZE)
      start = time.time()

      # train the discriminator
      for critic_itr in range(n_critic):
         idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
         batch_z      = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         batch_y      = train_annots[idx]
         batch_images = train_images[idx]

         sess.run(D_train_op, feed_dict={z:batch_z, y:batch_y, real_images:batch_images})
      
      # now train the generator once! use normal distribution, not uniform!!
      idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
      batch_z      = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
      batch_y      = train_annots[idx]
      batch_images = train_images[idx]

      # now get all losses and summary *without* performing a training step - for tensorboard and printing
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
         batch_z      = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         batch_y      = test_annots[idx]
         batch_images = test_images[idx]
         batch_ids    = test_ids[idx]
         gen_imgs = np.squeeze(np.asarray(sess.run([gen_images], feed_dict={z:batch_z, y:batch_y, real_images:batch_images})))

         num = 0
         for img,atr in zip(gen_imgs, batch_y):
            img = (img+1.)
            img *= 127.5
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = np.reshape(img, (SIZE, SIZE, -1))
            misc.imsave(IMAGES_DIR+'step_'+str(step)+'_'+str(batch_ids[num])+'.png', img)
            with open(IMAGES_DIR+'attrs.txt', 'a') as f:
               f.write('step_'+str(step)+'_'+str(batch_ids[num])+','+str(atr)+'\n')
            num += 1
            if num == 5: break
   saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
   saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')


