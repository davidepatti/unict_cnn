# (1) Importing dependency
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

import sys

seed = sys.argv[1]
np.random.seed(int(seed))

# (2) Get Data
import tflearn.datasets.oxflower17 as oxflower17
x, y = oxflower17.load_data(one_hot=True)

########################################
#
# (1) Create Training (80%), test (20%) and validation (20%) dataset
#     Datasets (x and y) are loaded as numpy object from the previous step
x_train, x_test_pre, y_train, y_test_pre = train_test_split(x, y, test_size=0.50, random_state=42)
x_test, x_validation, y_test, y_validation = train_test_split(x_test_pre, y_test_pre, test_size=0.5)


pickle.dump([x_train, x_test_pre, y_train, y_test_pre],open("train_"+seed+".data","wb"))
pickle.dump([x_test, x_validation, y_test, y_validation],open("test_"+seed+".data","wb"))

#
# Check Shapes
print('Shape: x_train={}, y_train={}'.format(x_train.shape, y_train.shape))
print('Shape: x_test={}, y_test={}'.format(x_test.shape, y_test.shape))
print('Shape: x_validation={}, y_validation={}'.format(x_validation.shape, y_validation.shape))
#####################################





