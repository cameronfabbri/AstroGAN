'''

   GalaxyZoo attribute config file. This decides which attributes will
   be used in the conditional GAN. There are a total of 37 attributes
   that correspond to questions in a decision tree, seen below.

   https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge#the-galaxy-zoo-decision-tree
   https://data.galaxyzoo.org/gz_trees/gz_trees.html

   Flags are binary - 1 to use an attribute, 0 to not use it

   The use_all flag overrides all and uses every attribute.

'''

import numpy as np

# Question 1: Is the object a smooth galaxy, a galaxy with features/disk or a star?
class1_1=1
class1_2=1
class1_3=1

# Question 2: Is it edge-on?
class2_1=1
class2_2=1

# Question 3: Is there a bar?
class3_1=1
class3_2=1

# Question 4: Is there a spiral pattern?
class4_1=0
class4_2=0

# Question 5: How prominent is the central bulge?
class5_1=0
class5_2=0
class5_3=0
class5_4=0

# Question 6: Is there anything "odd" about the galaxy?
class6_1=0
class6_2=0

# Question 7: How round is the smooth galaxy?
class7_1=0
class7_2=0
class7_3=0

# Question 8: What is the odd feature? 
class8_1=0
class8_2=0
class8_3=0
class8_4=0
class8_5=0
class8_6=0
class8_7=0

# Question 9: What shape is the bulge in the edge-on galaxy?
class9_1=0
class9_2=0
class9_3=0

# Question 10: How tightly wound are the spiral arms?
class10_1=0
class10_2=0
class10_3=0

# Question 11: How many spiral arms are there?
class11_1=0
class11_2=0
class11_3=0
class11_4=0
class11_5=0
class11_6=0

# all of the classes in an array. This will be used as a mask on the attributes loaded.
classes = np.asarray([class1_1,  class1_2,  class1_3,
                      class2_1,  class2_2,
                      class3_1,  class3_2,
                      class4_1,  class4_2,
                      class5_1,  class5_2,  class5_3,  class5_4,
                      class6_1,  class6_2,
                      class7_1,  class7_2,  class7_3,
                      class8_1,  class8_2,  class8_3,  class8_4, class8_5, class8_6, class8_7,
                      class9_1,  class9_2,  class9_3,
                      class10_1, class10_2, class10_3,
                      class11_1, class11_2, class11_3, class11_4, class11_5, class11_6])
