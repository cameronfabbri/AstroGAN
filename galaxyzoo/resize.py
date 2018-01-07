'''

   The output from the GAN is 64x64, so this just resizes them
   to be the size for GalaxyZoo, 424x424

'''

import scipy.misc as misc
from glob import glob
from tqdm import tqdm
import sys

DIR = sys.argv[1]

for img in tqdm(glob(DIR+'*.png')):
   img_ = misc.imread(img)
   img_ = misc.imresize(img_, (424,424))
   misc.imsave(img, img_)
