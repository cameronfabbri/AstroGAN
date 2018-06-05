import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tf_ops import *

'''
   Generator network
   batch norm before activation function
'''
def netG(z, y, UPSAMPLE):

   # concat attribute y onto z
   z = tf.concat([z,y], axis=1)
   z = tcl.fully_connected(z, 4*4*512, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_z')
   z = tf.reshape(z, [-1, 4, 4, 512])
   # their implementation has this just linear
   #z = tcl.batch_norm(z)
   #z = tf.nn.relu(z)
   print 'z:',z

   if UPSAMPLE == 'transpose':
      conv = tcl.convolution2d_transpose(z, 512, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv1')
   else:
      conv = upconv2d(z, 512, method=UPSAMPLE, kernel_size=5, name='g_conv1')
   print 'conv1:',conv
   
   if UPSAMPLE == 'transpose':
      conv = tcl.convolution2d_transpose(conv, 256, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv2')
   else:
      conv = upconv2d(conv, 256, method=UPSAMPLE, kernel_size=5, name='g_conv2')
   print 'conv2:',conv
   
   if UPSAMPLE == 'transpose':
      conv = tcl.convolution2d_transpose(conv, 128, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv3')
   else:
      conv = upconv2d(conv, 128, method=UPSAMPLE, kernel_size=5, name='g_conv3')
   print 'conv3:',conv

   if UPSAMPLE == 'transpose':
      conv = tcl.convolution2d_transpose(conv, 3, 5, 2, activation_fn=tf.nn.tanh, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv5')
   else:
      conv = upconv2d(conv, 3, method=UPSAMPLE, kernel_size=5, name='g_conv4')
   print 'conv5:',conv
   
   print
   print 'END G'
   print
   return conv


'''
   Discriminator network. No batch norm when using WGAN
'''
def netD(input_images, y, GAN, SIZE, PREDICT, reuse=False):

   print 'DISCRIMINATOR reuse = '+str(reuse)
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      y_dim = int(y.get_shape().as_list()[-1])

      # reshape so it's batchx1x1xy_size
      y = tf.reshape(y, shape=[-1, 1, 1, y_dim])
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

      # if true, also predict the morphology
      if PREDICT: 
         fc = tcl.fully_connected(tcl.flatten(conv), 37, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_fc')

         return conv, fc

      print 'END D\n'
      return conv, None

def netGResnet(z, y, SIZE):
   
   z = tf.concat([z,y], axis=1)
   z = tcl.fully_connected(z, 4*4*256, activation_fn=tf.identity, scope='g_z')
   z = tf.reshape(z, [-1, 4, 4, 256])
   z = tcl.batch_norm(z)
   z = tf.nn.relu(z)
   print 'z:',z
   print

   conv1 = resBlockUp(z, 1, 'g')
   print 'conv1:',conv1
   print
   conv2 = resBlockUp(conv1, 2, 'g')
   print 'conv2:',conv2
   print
   conv3 = resBlockUp(conv2, 3, 'g')
   print 'conv3:',conv3
   print
   conv4 = resBlockUp(conv3, 4, 'g')
   print 'conv4:',conv4
   print
   
   conv5 = tcl.conv2d(conv4, 3, 3, 1, activation_fn=tf.nn.tanh, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv5')
   print 'conv5:',conv5

   return conv5

def netDResnet(input_images, y, GAN, SIZE, reuse=False):
   
   print 'DISCRIMINATOR reuse = '+str(reuse)
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):
      y_dim = int(y.get_shape().as_list()[-1])
      # reshape so it's batchx1x1xy_size
      y = tf.reshape(y, shape=[-1, 1, 1, y_dim])
      print 'input_images:',input_images
      print
      input_ = conv_cond_concat(input_images, y)
      
      conv1 = tcl.conv2d(input_, 64, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')

      res = resBlockDown(conv1, 1, 'd')
      print 'res:',res
      print
      res = resBlockDown(res, 2, 'd')
      print 'res:',res
      print
      res = resBlockDown(res, 3, 'd')
      print 'res:',res
      print
      #res = resBlockDown(res, 4, 'd')
      #print 'res:',res
      #print

      return res


def resBlock(x, num, n):

   x = tf.nn.relu(x)

   conv1 = tcl.conv2d(x, 64, 3, 1, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope=n+'_resconv1_'+str(num))
   conv1 = tf.nn.relu(conv1)
   print 'res_conv1:',conv1

   conv2 = tcl.conv2d(conv1, 64, 3, 1, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope=n+'_resconv2_'+str(num))
   print 'res_conv2:',conv2
   
   output = tf.add(x,conv2)
   print 'res_out:',output
   return output

def resBlockUp(x, num, n):

   shapes = x.get_shape().as_list()
   height = shapes[1]
   width  = shapes[2]
   x = tf.image.resize_nearest_neighbor(x, [height*2, width*2])
   x = tf.nn.relu(x)

   conv1 = tcl.conv2d(x, 256, 3, 1, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope=n+'_resconv1_'+str(num))
   conv1 = tf.nn.relu(conv1)
   print 'res_conv1:',conv1

   conv2 = tcl.conv2d(conv1, 256, 3, 1, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope=n+'_resconv2_'+str(num))
   print 'res_conv2:',conv2
   
   output = tf.add(x,conv2)
   print 'res_out:',output
   return output

def resBlockDown(x, num, n):

   x = tcl.conv2d(x, 64, 3, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope=n+'_resconv0_'+str(num))
   x = tf.nn.relu(x)

   conv1 = tcl.conv2d(x, 64, 3, 1, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope=n+'_resconv1_'+str(num))
   conv1 = tf.nn.relu(conv1)
   print 'res_conv1:',conv1

   conv2 = tcl.conv2d(conv1, 64, 3, 1, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope=n+'_resconv2_'+str(num))
   print 'res_conv2:',conv2
   
   output = tf.add(x,conv2)
   print 'res_out:',output
   return output


def activate(x, ACTIVATION):
   if ACTIVATION == 'lrelu': return lrelu(x)
   if ACTIVATION == 'relu':  return relu(x)
   if ACTIVATION == 'elu':   return elu(x)
   if ACTIVATION == 'swish': return swish(x)
