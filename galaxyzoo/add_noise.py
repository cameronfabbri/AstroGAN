import numpy as np
import scipy.misc as misc

original = misc.imread('0.png').astype(float)
noise    = np.random.poisson(original)

#misc.imsave('noise1.png', noise)

new = original+noise
misc.imsave('1.png', new)



'''
width, height, c = noise.shape

for i in range(width):
   for j in range(height):
      if i%4 != 0 and j % 4 != 0:
         noise[i,j,:] = np.asarray([0,0,0])
'''

