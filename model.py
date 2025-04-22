
import tensorflow as tf
import numpy as np

def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(43, activation='softmax')
    ])
    return model

def load_weights(model):
    weights = np.load("model_weights.npz", allow_pickle=True)
    model.set_weights(weights['arr_0'])
    return model
