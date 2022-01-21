# (1) Importing dependency
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import os
import pickle
import sys

if len(sys.argv)==1:
    print("please specify seed!")
    exit(-1)

seed = sys.argv[1]
np.random.seed(int(seed))

# (2) Get Data
import tflearn.datasets.oxflower17 as oxflower17
x, y = oxflower17.load_data(one_hot=True)

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

model.summary()

# (4) Compile 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# checkpoint
filepath = os.path.join("checkpoint", "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# (5) Train
#model.fit(x, y, batch_size=64, epochs=200, verbose=1,  validation_split=0.2, shuffle=True)
model.fit(x, y, batch_size=40, epochs=100, verbose=1, callbacks=callbacks_list, validation_split=0.5, shuffle=True)

model.save_weights('alexnet_100.h5')
model.load_weights('alexnet_100.h5')


score = model.evaluate(x_validation, y_validation)
print('Test loss:', score[0])
print('Test accuracy:', score[1])





