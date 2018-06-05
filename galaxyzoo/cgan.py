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

# prints the entire numpy array
#np.set_printoptions(threshold=np.nan)
# my own imports
sys.path.insert(0, '../ops/')
sys.path.insert(0, '../')
from config import classes
import data_ops
from nets import *
from tf_ops import *

if __name__ == '__main__':

   # params
   parser = argparse.ArgumentParser()
   parser.add_argument('--GAN',        required=False,help='Type of GAN loss to use',  type=str,  default='wgan')
   parser.add_argument('--CROP',       required=False,help='Center crop images or not',type=int,  default=0)
   parser.add_argument('--SIZE',       required=False,help='Output size of generator', type=int,  default=64)
   parser.add_argument('--BETA1',      required=False,help='beta1 ADAM parameter',     type=float,default=0.)
   parser.add_argument('--EPOCHS',     required=False,help='Maximum number of epochs', type=int,  default=100)
   parser.add_argument('--NETWORK',    required=False,help='Network to use',           type=str,  default='dcgan')
   parser.add_argument('--PREDICT',    required=False,help='D predicts morphology',    type=int,  default=0)
   parser.add_argument('--UPSAMPLE',   required=False,help='Method to upsample in G',  type=str,  default='transpose')
   parser.add_argument('--DATA_DIR',   required=True, help='Directory where data is',  type=str,  default='./')
   parser.add_argument('--BATCH_SIZE', required=False,help='Batch size',               type=int,  default=64)
   a = parser.parse_args()

   GAN            = a.GAN
   SIZE           = a.SIZE
   CROP           = bool(a.CROP)
   BETA1          = a.BETA1
   EPOCHS         = a.EPOCHS
   PREDICT        = bool(a.PREDICT)
   NETWORK        = a.NETWORK
   DATA_DIR       = a.DATA_DIR
   UPSAMPLE       = a.UPSAMPLE
   BATCH_SIZE     = a.BATCH_SIZE

   # convert to string for directory naming
   cn = ''
   for i in classes:
      cn = cn + str(i)

   CHECKPOINT_DIR = 'checkpoints/GAN_'+GAN\
                    +'/UPSAMPLE_'+str(UPSAMPLE)\
                    +'/BETA1_'+str(BETA1)\
                    +'/CLASSES_'+str(cn)\
                    +'/NETWORK_'+NETWORK\
                    +'/CROP_'+str(CROP)\
                    +'/PREDICT_'+str(PREDICT)\
                    +'/SIZE_'+str(SIZE)\
                    +'/'

   IMAGES_DIR     = CHECKPOINT_DIR+'images/'

   # store all this information in a pickle file
   info_dict = {}
   info_dict['GAN']            = GAN
   info_dict['SIZE']           = SIZE
   info_dict['CROP']           = CROP
   info_dict['BETA1']          = BETA1
   info_dict['NETWORK']        = NETWORK
   info_dict['CLASSES']        = classes
   info_dict['UPSAMPLE']       = UPSAMPLE
   info_dict['DATA_DIR']       = DATA_DIR
   info_dict['BATCH_SIZE']     = BATCH_SIZE
   info_dict['CHECKPOINT_DIR'] = CHECKPOINT_DIR

   try: os.makedirs(IMAGES_DIR)
   except: pass

   exp_pkl = open(CHECKPOINT_DIR+'info.pkl', 'wb')
   data = pickle.dumps(info_dict)
   exp_pkl.write(data)
   exp_pkl.close()

   global_step = tf.Variable(0, name='global_step', trainable=False)
   # placeholders for data going into the network
   real_images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, SIZE, SIZE, 3), name='real_images')
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 128), name='z')
   y           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 37), name='y')
   mask        = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 37), name='mask')

   # multiply y by the mask of attributes actually being used
   y = tf.multiply(y,mask)

   # repeat the classes mask to be of batch size
   classes = np.array([classes,]*BATCH_SIZE)

   # generated images
   if NETWORK == 'dcgan': gen_images = netG(z, y, UPSAMPLE)
   if NETWORK == 'resnet': gen_images = netGResnet(z, y, UPSAMPLE)

   # get the output from D on the real and fake data
   if NETWORK == 'dcgan': errD_real, pred_real = netD(real_images, y, GAN, SIZE, PREDICT)
   if NETWORK == 'dcgan': errD_fake, pred_fake = netD(gen_images, y, GAN, SIZE, PREDICT, reuse=True)
   
   if NETWORK == 'resnet': errD_real = netDResnet(real_images, y, GAN, SIZE)
   if NETWORK == 'resnet': errD_fake = netDResnet(gen_images, y, GAN, SIZE, reuse=True)

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
      if NETWORK == 'dcgan': d_hat, m = netD(x_hat, y, GAN, SIZE, PREDICT, reuse=True)
      if NETWORK == 'resnet': d_hat = netDResnet(x_hat, y, GAN, SIZE, reuse=True)
      gradients = tf.gradients(d_hat, x_hat)[0]
      slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
      gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
      errD += gradient_penalty
      
      # training details
      n_critic = 5
      beta1    = BETA1
      beta2    = 0.9
      lr       = 1e-4

   if PREDICT:
      print 'Using D as a prediction network as well...'
      errP1 = tf.nn.l2_loss(y-pred_real)
      errP2 = tf.nn.l2_loss(y-pred_fake)
      errP = (errP1+errP2)/2.

      errG = errG + errP

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   try: tf.summary.scalar('predict_loss', errP)
   except: pass
   merged_summary_op = tf.summary.merge_all()

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   G_train_op = tf.train.AdamOptimizer(learning_rate=lr,beta1=beta1,beta2=beta2).minimize(errG, var_list=g_vars, global_step=global_step)
   D_train_op = tf.train.AdamOptimizer(learning_rate=lr,beta1=beta1,beta2=beta2).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=1)
   init  = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
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
   train_images, train_annots, train_ids, test_images, test_annots, test_ids = data_ops.load_zoo(DATA_DIR, SIZE, crop=CROP)
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
         batch_z      = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 128]).astype(np.float32)
         batch_y      = train_annots[idx]
         batch_images = train_images[idx]

         # randomly flip images left right or up down
         r = random.random()
         if r < 0.5: batch_images = np.fliplr(batch_images)

         r = random.random()
         if r < 0.5: batch_images = np.flipud(batch_images)

         sess.run(D_train_op, feed_dict={z:batch_z, y:batch_y, real_images:batch_images, mask:classes})
      
      # now train the generator once! use normal distribution, not uniform!!
      idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
      batch_z      = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 128]).astype(np.float32)
      batch_y      = train_annots[idx]
      batch_images = train_images[idx]

      # now get all losses and summary *without* performing a training step - for tensorboard and printing
      sess.run(G_train_op, feed_dict={z:batch_z, y:batch_y, real_images:batch_images, mask:classes})
      if PREDICT:
         D_loss, G_loss, P_loss, summary = sess.run([errD, errG, errP, merged_summary_op],
                                 feed_dict={z:batch_z, y:batch_y, real_images:batch_images, mask:classes})
      else:
         D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op],
                                 feed_dict={z:batch_z, y:batch_y, real_images:batch_images, mask:classes})

      summary_writer.add_summary(summary, step)

      try: print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'P_loss:',P_loss
      except: print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss
      step += 1
    
      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')

         idx          = np.random.choice(np.arange(test_len), BATCH_SIZE, replace=False)
         batch_z      = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 128]).astype(np.float32)
         batch_y      = test_annots[idx]
         batch_images = test_images[idx]
         batch_ids    = test_ids[idx]
         gen_imgs = np.squeeze(np.asarray(sess.run([gen_images], feed_dict={z:batch_z, y:batch_y, real_images:batch_images, mask:classes})))

         num = 0
         # gotta multiply by mask here
         for img,atr in zip(gen_imgs, np.multiply(batch_y,classes)):
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


