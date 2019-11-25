import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.utils import CustomObjectScope

def softmax(x):
    x = x * 0.0018394317012280226
    return tf.keras.activations.softmax(x)


def qt_InceptionV3(weights): 
    with CustomObjectScope({'softmax': softmax}): 
        json_string = open("InceptionV3.json", "r").read()
        model = model_from_json(json_string) 
        model.load_weights(weights)
        return model
