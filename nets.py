import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tf_ops import *

'''
   Generator network
   batch norm before activation function
'''
def netG(z, y, BATCH_SIZE, SIZE):

   # concat attribute y onto z
   z = tf.concat([z,y], axis=1)
   z = tcl.fully_connected(z, 4*4*512, activation_fn=tf.identity, scope='g_z')
   z = tf.reshape(z, [BATCH_SIZE, 4, 4, 512])
   z = tcl.batch_norm(z)
   z = tf.nn.relu(z)
   print 'z:',z

   conv = tcl.convolution2d_transpose(z, 512, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv1')
   print 'conv1:',conv
   
   conv = tcl.convolution2d_transpose(conv, 256, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv2')
   print 'conv2:',conv
   
   conv = tcl.convolution2d_transpose(conv, 128, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv3')
   print 'conv3:',conv

   if SIZE == 128:
      conv = tcl.convolution2d_transpose(conv, 64, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv4')
      print 'conv4:',conv
   
   conv = tcl.convolution2d_transpose(conv, 3, 5, 2, activation_fn=tf.nn.tanh, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv5')
   print 'conv5:',conv
   
   print
   print 'END G'
   print
   return conv

'''
   Discriminator network. No batch norm when using WGAN
'''
def netD(input_images, y, BATCH_SIZE, GAN, SIZE, reuse=False):

   print 'DISCRIMINATOR reuse = '+str(reuse)
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      y_dim = int(y.get_shape().as_list()[-1])

      # reshape so it's batchx1x1xy_size
      y = tf.reshape(y, shape=[BATCH_SIZE, 1, 1, y_dim])
      print 'input_images:',input_images
      input_ = conv_cond_concat(input_images, y)

      conv = tcl.conv2d(input_, 64, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
      conv = lrelu(conv)
      print 'conv1:',conv
      
      conv = tcl.conv2d(conv, 128, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv2')
      if GAN != 'wgan': conv = tcl.batch_norm(conv)
      conv = lrelu(conv)
      print 'conv2:',conv

      conv = tcl.conv2d(conv, 256, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv3')
      if GAN != 'wgan': conv = tcl.batch_norm(conv)
      conv = lrelu(conv)
      print 'conv3:',conv

      conv = tcl.conv2d(conv, 512, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv4')
      if GAN != 'wgan': conv = tcl.batch_norm(conv)
      conv = lrelu(conv)
      print 'conv4:',conv

      if SIZE == 128:
         conv = tcl.conv2d(conv, 512, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv5')
         if GAN != 'wgan': conv = tcl.batch_norm(conv)
         conv = lrelu(conv)
         print 'conv5:',conv

      conv = tcl.conv2d(conv, 1, 4, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv6')
      print 'conv6:',conv

      print 'input images:',input_images
      print 'END D\n'
      return conv

def activate(x, ACTIVATION):
   if ACTIVATION == 'lrelu': return lrelu(x)
   if ACTIVATION == 'relu':  return relu(x)
   if ACTIVATION == 'elu':   return elu(x)
   if ACTIVATION == 'swish': return swish(x)
