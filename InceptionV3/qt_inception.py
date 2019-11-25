import sys
if len(sys.argv) < 3:
    print('Usage: '+sys.argv[0]+' <h5file> <input>')
    exit(0)
#from keras.applications.inception_v3 import InceptionV3
from hybrid_inceptionv3 import qt_InceptionV3

from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.models import Model
from keras.optimizers import SGD
import numpy as np
import cv2
from keras.layers import Input

weights_path = sys.argv[1]

#model = MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights=weights_path, input_tensor=None, pooling=None, classes=1000)

model = qt_InceptionV3(weights_path)
#model = InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=(299,299,3), pooling=None, classes=1000)

#Alternative custom input shape
#input_tensor = Input(shape=(224, 224, 3))
#model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)


if __name__ == "__main__":

    if sys.argv[2] == 'summary':
        model.summary()
        exit(0)

    img_path = sys.argv[2]
    img = image.load_img(img_path, target_size=(299, 299))
    #img = image.load_img(img_path, target_size=(224, 224))
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
		#target_size=(224, 224), 
		target_size=(299, 299), 
		batch_size=10,
		class_mode='categorical',
		shuffle=False)

    model.trainable=False
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy', in_top_k])

    results = model.evaluate_generator(validation_generator, steps=1000, workers=1, max_queue_size=1)
    #results = model.evaluate_generator(validation_generator, steps=5000, workers=1, max_queue_size=1)

    print('--> Results for '+sys.argv[1])
    print(model.metrics_names)
    print(results)

#########################################


