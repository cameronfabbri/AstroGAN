'''

   Collecting stats on the efigi dataset

'''

f_ = '/mnt/data1/images/efigi/EFIGI_attributes.txt'

import numpy as np

T = []

with open(f_,'r') as f:
   for line in f:
      line     = line.rstrip().split()
      image_id = line[0]
      line     = np.asarray([float(x) for x in line[1:]])
      T.append(line[0])

print sorted(set(T))

