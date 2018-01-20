'''

   Data management stuff

'''

import scipy.misc as misc
import cPickle as pickle
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import requests
import ntpath
import math
import glob
import gzip
import os


'''
   Image normalizing functions
'''
def normalize(image):
   return (image/127.5)-1.0

def unnormalize(image):
   img = (img+1.)
   img *= 127.5
   return img

'''

   Loading up the galaxy dataset.
   
   1  T               - EFIGI morphological type
   7  Arm strength    - Strength of spiral arms
   10 Arm curvature   - Average curvature of the spiral arms
   13 Arm Rotation    - Direction of the winding of the spiral arms
   16 Bar length      - Length of the central bar
   19 Inner Ring      - Strength of the inner ring, lens or inner pseudo-ring
   22 Outer Ring      - Strength of outer ring
   25 Pseudo Ring     - Type and strength of outer pseudo-ring
   28 Perturbation    - Deviation from rotational symmetry
   31 Visible Dust    - Strength of dust features
   34 Dust Dispersion - Patchiness of dust features
   40 Hot Spots       - Strength of regions of strong star formation, active nuclei, or stellar nuclie
   49 Multiplicity    - Abundance of neighbouring galaxies

'''
def load_efigi(data_dir, redshift, size):

   redict = {}
   if redshift:
      d=0
      with open(data_dir+'EFIGI_coord_redshift.txt','r') as f:
         for line in f:
            if d==0:
               d=1
               continue
            line = line.rstrip().split()
            galaxy_id = line[0]
            redshift  = float(line[9])
            redict[galaxy_id] = redshift

   #idx = np.array([0, 1, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 40, 49])
   idx = np.array([7, 10, 31, 49])

   train_images     = glob.glob(data_dir+'images/train/*.png')
   test_images      = glob.glob(data_dir+'images/test/*.png')

   # get train ids from train folder
   train_ids = [ntpath.basename(x.split('.')[0]) for x in train_images]
   test_ids  = [ntpath.basename(x.split('.')[0]) for x in test_images]

   iptr = data_dir+'images/train/'
   ipte = data_dir+'images/test/'

   train_images = []
   test_images  = []
   train_attributes = []
   test_attributes  = []

   paths = []
   with open(data_dir+'EFIGI_attributes.txt', 'r') as f:
      for line in f:
         line     = line.rstrip().split()
         galaxy_id = line[0]
         line     = np.asarray(line[1:])
         line     = line[idx].astype('float32')
         if redshift: line = np.append(line, redict[galaxy_id]) # if using redshift, add it to the attributes
         
         if galaxy_id in train_ids:
            img = misc.imread(iptr+galaxy_id+'.png').astype('float32')
            img = misc.imresize(img, (size, size))
            img = normalize(img)
            train_images.append(img)
            train_attributes.append(line)
         elif galaxy_id in test_ids:
            paths.append(ipte+galaxy_id+'.png')
            img = misc.imread(ipte+galaxy_id+'.png').astype('float32')
            img = misc.imresize(img, (size, size))
            img = normalize(img)
            test_images.append(img)
            test_attributes.append(line)

   return np.asarray(train_images), np.asarray(train_attributes), np.asarray(train_ids), np.asarray(test_images), np.asarray(test_attributes), np.asarray(test_ids)

'''
   Galaxy zoo dataset.
'''
def load_zoo(data_dir, size):

   train_images = sorted(glob.glob(data_dir+'images_training_rev1/train/*.jpg'))
   train_ids    = [ntpath.basename(x.split('.')[0]) for x in train_images]
   
   test_images = sorted(glob.glob(data_dir+'images_training_rev1/test/*.jpg'))
   test_ids    = [ntpath.basename(x.split('.')[0]) for x in test_images]

   # put ids with image path just to be sure we get the correct image. Slow but oh well, only done once
   id_dict = {}
   for img, id_ in zip(train_images, train_ids):
      id_dict[int(id_)] = img
   for img, id_ in zip(test_images, test_ids):
      id_dict[int(id_)] = img

   train_images     = []
   train_attributes = []

   test_images     = []
   test_attributes = []

   d = 0
   with open(data_dir+'training_solutions_rev1.csv', 'r') as f:
      for line in f:
         if d == 0:
            d = 1
            continue
         line = np.asarray(line.split(',')).astype('float32')
         im_id = int(line[0])
         img = misc.imread(id_dict[im_id]).astype('float32')
         img = misc.imresize(img, (size,size))
         img = normalize(img)
         att = line[1:]

         # remember train_ids is all str
         if str(im_id) in train_ids:
            train_images.append(img)
            train_attributes.append(att)
         else:
            test_images.append(img)
            test_attributes.append(att)

         d += 1
         #if d == 65: break

   return np.asarray(train_images), np.asarray(train_attributes), np.asarray(train_ids), np.asarray(test_images), np.asarray(test_attributes), np.asarray(test_ids)


