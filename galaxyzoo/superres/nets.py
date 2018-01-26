import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tf_ops import *

'''
   Generator network

   Input: 64x64x3
   Output: 256x256x3
'''
def netG(x):

   print 'x:',x
   enc_conv1 = tcl.conv2d(x, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv1')
   enc_conv1 = tcl.batch_norm(enc_conv1)
   enc_conv1 = lrelu(enc_conv1)
   print 'enc_conv1:',enc_conv1

   enc_conv2 = tcl.conv2d(enc_conv1, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv2')
   enc_conv2 = tcl.batch_norm(enc_conv2)
   enc_conv2 = lrelu(enc_conv2)
   print 'enc_conv2:',enc_conv2
   
   enc_conv3 = tcl.conv2d(enc_conv2, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv3')
   enc_conv3 = tcl.batch_norm(enc_conv3)
   enc_conv3 = lrelu(enc_conv3)
   print 'enc_conv3:',enc_conv3

   enc_conv4 = tcl.conv2d(enc_conv3, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv4')
   enc_conv4 = tcl.batch_norm(enc_conv4)
   enc_conv4 = lrelu(enc_conv4)
   print 'enc_conv4:',enc_conv4

   enc_conv5 = tcl.conv2d(enc_conv4, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv5')
   enc_conv5 = tcl.batch_norm(enc_conv5)
   enc_conv5 = lrelu(enc_conv5)
   print 'enc_conv5:',enc_conv5
   
   enc_conv6 = tcl.conv2d(enc_conv5, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv6')
   enc_conv6 = tcl.batch_norm(enc_conv6)
   enc_conv6 = lrelu(enc_conv6)
   print 'enc_conv6:',enc_conv6

   # decoder, no batch norm
   dec_conv1 = tcl.convolution2d_transpose(enc_conv6, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv1')
   dec_conv1 = relu(dec_conv1)
   #dec_conv1 = tf.concat([dec_conv1, enc_conv3], axis=3)
   print 'dec_conv1:',dec_conv1

   dec_conv2 = tcl.convolution2d_transpose(dec_conv1, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv2')
   dec_conv2 = relu(dec_conv2)
   #dec_conv2 = tf.concat([dec_conv2, enc_conv2], axis=3)
   print 'dec_conv2:',dec_conv2
   
   dec_conv3 = tcl.convolution2d_transpose(dec_conv2, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv3')
   dec_conv3 = relu(dec_conv3)
   #dec_conv3 = tf.concat([dec_conv3, enc_conv1], axis=3)
   print 'dec_conv3:',dec_conv3

   dec_conv4 = tcl.convolution2d_transpose(dec_conv3, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv4')
   dec_conv4 = relu(dec_conv4)
   #dec_conv4 = tf.concat([dec_conv4, x], axis=3)
   print 'dec_conv4:',dec_conv4
   
   dec_conv5 = tcl.convolution2d_transpose(dec_conv4, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv5')
   dec_conv5 = relu(dec_conv5)
   #dec_conv5 = tf.concat([dec_conv5, enc_conv3], axis=3)
   print 'dec_conv5:',dec_conv5

   dec_conv6 = tcl.convolution2d_transpose(dec_conv5, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv6')
   dec_conv6 = relu(dec_conv6)
   #dec_conv6 = tf.concat([dec_conv6, enc_conv2], axis=3)
   print 'dec_conv6:',dec_conv6
   
   dec_conv7 = tcl.convolution2d_transpose(dec_conv6, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv7')
   dec_conv7 = relu(dec_conv7)
   #dec_conv7 = tf.concat([dec_conv7, enc_conv1], axis=3)
   print 'dec_conv7:',dec_conv7
   
   dec_conv8 = tcl.convolution2d_transpose(dec_conv7, 3, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv8')
   dec_conv8 = tanh(dec_conv8)
   print 'dec_conv8', dec_conv8

   return dec_conv8




def netD(x, reuse=False):
   print
   print 'netD'

   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      conv1 = tcl.conv2d(x, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
      conv1 = lrelu(conv1)
      
      conv2 = tcl.conv2d(conv1, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv2')
      conv2 = lrelu(conv2)
      
      conv3 = tcl.conv2d(conv2, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv3')
      conv3 = lrelu(conv3)
      
      conv4 = tcl.conv2d(conv3, 512, 4, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv4')
      conv4 = lrelu(conv4)
      
      conv5 = tcl.conv2d(conv4, 1, 1, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv5')

      print 'x:',x
      print 'conv1:',conv1
      print 'conv2:',conv2
      print 'conv3:',conv3
      print 'conv4:',conv4
      print 'conv5:',conv5
      return conv5

