import sys

if len(sys.argv) < 3:
    print('Usage: '+sys.argv[0]+' <h5file> <input>')
    exit(0)

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import cv2, numpy as np

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=(224, 224, 3), data_format='channels_last'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":

    print('--> Testing inference...')

    # Test pretrained model
    model = VGG_16(sys.argv[1])
    if sys.argv[2] == 'summary':
        model.summary()
        exit(0)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    img_path = sys.argv[2]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print(decode_predictions(preds))

#########################################
    print('--> Starting evalutation...')
    from keras.preprocessing.image import ImageDataGenerator
    from keras import metrics
    
    def in_top_k(y_true, y_pred):
        return metrics.top_k_categorical_accuracy(y_true,y_pred,k=5)

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    validation_generator = val_datagen.flow_from_directory('./imagenet-data/validation',
		target_size=(224, 224), 
		batch_size=1,
		class_mode='categorical',
		shuffle=False)

    model.trainable=False
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy', in_top_k])

    results = model.evaluate_generator(validation_generator, steps=5, workers=1, max_queue_size=1)

    print('--> Results:')
    print(model.metrics_names)
    print(results)

#########################################
