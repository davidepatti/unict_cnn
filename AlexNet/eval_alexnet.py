import sys
if len(sys.argv) < 3:
    print('Need to pass h5 file and seed')
    exit(0)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics
import numpy as np
np.random.seed(int(sys.argv[2]))

# (2) Get Data
#import tflearn.datasets.oxflower17 as oxflower17
#x, y = oxflower17.load_data(one_hot=True)

########################################
#
# (1) Create Training (80%), test (20%) and validation (20%) dataset
#     Datasets (x and y) are loaded as numpy object from the previous step
#x_train, x_test_pre, y_train, y_test_pre = train_test_split(x, y, test_size=0.20, random_state=42)
#x_train, x_test_pre, y_train, y_test_pre = train_test_split(x, y, test_size=0.50, random_state=42)
#x_test, x_validation, y_test, y_validation = train_test_split(x_test_pre, y_test_pre, test_size=0.5)

import pickle
seed = sys.argv[2]
[x_train, x_test_pre, y_train, y_test_pre]= pickle.load(open("train_"+seed+".data","rb"))
[x_test, x_validation, y_test, y_validation]= pickle.load(open("test_"+seed+".data","rb"))
#
# Check Shapes
print('Shape: x_train={}, y_train={}'.format(x_train.shape, y_train.shape))
print('Shape: x_test={}, y_test={}'.format(x_test.shape, y_test.shape))
print('Shape: x_validation={}, y_validation={}'.format(x_validation.shape, y_validation.shape))
#####################################

# (3) Create a sequential model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11),\
 strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.5))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.5))
# Batch Normalisation
model.add(BatchNormalization())

## 3rd Dense Layer
#model.add(Dense(1000))
#model.add(Activation('relu'))
## Add Dropout
#model.add(Dropout(0.4))
## Batch Normalisation
#model.add(BatchNormalization())

# Output Layer
model.add(Dense(17))
model.add(Activation('softmax'))

if sys.argv[1]=='summary':
    model.summary()
    exit(0)

def in_top_k(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=5)  

# (4) Compile 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',in_top_k])

model.load_weights(sys.argv[1])

score = model.evaluate(x_validation, y_validation)
print('h5: ',sys.argv[1])
print('Seed: ',sys.argv[2])
print('Test_loss: ', score[0])
print('Test_accuracy: ', score[1])
print('Test_accuracy_top: ', score[2])

#model.save_weights(sys.argv[1]+".h5")

