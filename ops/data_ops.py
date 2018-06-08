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

   0 GalaxyID
   1  T               - EFIGI morphological type
   4  Bulge_to_Total  - Bulge-to-total ratio
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
   37 Flocculence     - Strength of scattered HII regions
   40 Hot Spots       - Strength of regions of strong star formation, active nuclei, or stellar nuclie
   43 Inclination     - Inclination of disks or elongation of spheroids
   46 Contamination   - Severity of contamination by stars, galaxies or artifacts
   49 Multiplicity    - Abundance of neighbouring galaxies

'''
def load_efigi(data_dir, classes, size):

   # get redshift which is in a different file
   redict = {}
   d=0
   with open(data_dir+'EFIGI_coord_redshift.txt','r') as f:
      for line in f:
         if d==0:
            d=1
            continue
         line = line.rstrip().split()
         galaxy_id = line[0]
         redshift  = float(line[9])
         if redshift < 0: continue # redshift missing, so has a value of -99.99 we don't want
         redict[galaxy_id] = redshift

   # grab these ones from the file, then multiply by the mask that comes in (classes variable)
   idx_ = np.array([1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49])
   idx_ = np.multiply(classes[:-1], idx_)
   idx = [x for x in idx_ if x != 0]

   # og
   #idx = np.array([7, 10, 31, 49])

   train_images     = glob.glob(data_dir+'images/train/*.png')
   test_images      = glob.glob(data_dir+'images/test/*.png')
   print len(train_images)
   # get train ids from train folder
   train_ids = [ntpath.basename(x.split('.')[0]) for x in train_images]
   test_ids  = [ntpath.basename(x.split('.')[0]) for x in test_images]

   iptr = data_dir+'images/train/'
   ipte = data_dir+'images/test/'

   train_images = []
   test_images  = []
   train_attributes = []
   test_attributes  = []

   r_test_ids = []
   paths = []

   with open(data_dir+'EFIGI_attributes.txt', 'r') as f:
      for line in f:
         line     = line.rstrip().split()
         galaxy_id = line[0]
         line     = np.asarray(line[1:])
         line     = line[idx].astype('float32')
         # add in the redshift attribute
         try: line = np.append(line, redict[galaxy_id])
         except: continue # don't use this one in training if no redshift (about 400 total)
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
            r_test_ids.append(galaxy_id)

   return np.asarray(train_images), np.asarray(train_attributes), np.asarray(train_ids), np.asarray(test_images), np.asarray(test_attributes), np.asarray(r_test_ids)

def crop_center(img,cropx,cropy):
   y,x,c = img.shape
   startx = x//2-(cropx//2)
   starty = y//2-(cropy//2)    
   return img[starty:starty+cropy,startx:startx+cropx]

def load_zoo(data_dir, hot=False):

   train_paths = []
   test_paths  = []

   train_attributes = []
   test_attributes  = []

   train_attributes_f = open(data_dir+'images_training_rev1/train.csv')
   test_attributes_f  = open(data_dir+'images_training_rev1/test.csv')

   # get train paths
   for line in train_attributes_f:
      line  = line.rstrip()
      line_ = line.split(',')
      gid   = line_[0]
      train_paths.append(data_dir+'images_training_rev1/train/'+gid+'.jpg')
      attribute = np.asarray(line_[1:]).astype(float)
      # converts the array to 1 or 0
      if hot: attribute = np.where(attribute > 0.5, 1, 0)
      train_attributes.append(attribute)

   # get test paths
   for line in test_attributes_f:
      line  = line.rstrip()
      line_ = line.split(',')
      gid   = line_[0]
      test_paths.append(data_dir+'images_training_rev1/test/'+gid+'.jpg')
      attribute = np.asarray(line_[1:]).astype(float)
      # converts the array to 1 or 0
      if hot: attribute = np.where(attribute > 0.5, 1, 0)
      test_attributes.append(attribute)

   return np.asarray(train_paths), np.asarray(train_attributes), np.asarray(test_paths), np.asarray(test_attributes)

'''
   Galaxy zoo dataset.
def load_zoo(data_dir, size, crop=True):

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

   #train_images     = []
   train_attributes = []

   #test_images     = []
   test_attributes = []

   d = 0
   with open(data_dir+'training_solutions_rev1.csv', 'r') as f:
      for line in tqdm(f):
         if d == 0:
            d = 1
            continue
         line = np.asarray(line.split(',')).astype('float32')
         im_id = int(line[0])
         #img = misc.imread(id_dict[im_id]).astype('float32')
         #if crop: img = crop_center(img, 212, 212)
         #img = misc.imresize(img, (size,size))
         #img = normalize(img)
         att = line[1:]

         # remember train_ids is all str
         if str(im_id) in train_ids:
            #train_images.append(img)
            train_attributes.append(att)
         else:
            #test_images.append(img)
            test_attributes.append(att)

         d += 1
         #if d == 65: break

   #return np.asarray(train_images), np.asarray(train_attributes), np.asarray(train_ids), np.asarray(test_images), np.asarray(test_attributes), np.asarray(test_ids)

'''

