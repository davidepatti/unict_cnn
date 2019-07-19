from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = val_datagen.flow_from_directory(
		'/data/xincoder/imagenet/imagenet-datavalidation',
		target_size=(224, 224), 
		batch_size=10,
		class_mode='categorical',
		shuffle=False)

model = VGG16(weights='imagenet', include_top=True)
model.trainable=False
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc', metrics.top_k_categorical_accuracy])

results = model.evaluate_generator(validation_generator, steps=5000, workers=1, max_queue_size=1)
print(results)
