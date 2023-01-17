import enum
from tokenize import String
from typing import Tuple
import tensorflow as tf


class Model:
    def __init__(self, input_shape: Tuple, num_classes: int, weights: String = None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = weights
        self._model = None

    def model_details(self):
        print("input size : {}  classes : {} weights : {} ".format(
            self.input_shape, self.num_classes, self.weights))


class MobileNetV2(Model):
    def __init__(self, input_shape: Tuple, num_classes: int, weights: String = None):
        super().__init__(input_shape, num_classes, weights)
        self._model = tf.keras.applications.MobileNetV2(
            self.input_shape, classes=self.num_classes, weights=self.weights)


class SimpleCNN(Model):
    def __init__(self, input_shape: Tuple, num_classes: int, weights: String = None):
        super().__init__(input_shape, num_classes, weights)

        self._model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                32, 3, padding='same', input_shape=self.input_shape, activation='relu'),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.num_classes, activation='softmax'),
        ])

class KerasExpCNN(Model):
    def __init__(self, input_shape: Tuple, num_classes: int, weights: String = None):
        super().__init__(input_shape, num_classes, weights)

        self._model = tf.keras.models.Sequential([
        tf.keras.Input(shape=self.input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(self.num_classes, activation="softmax"),
    ])
        

class MNISTCNN(Model):
    def __init__(self, input_shape: Tuple, num_classes: int, weights: String = None):
        super().__init__(input_shape, num_classes, weights)

        self._model = tf.keras.models.Sequential([
        tf.keras.Input(shape=self.input_shape),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(28,activation='relu'),
        tf.keras.layers.Dense(self.num_classes, activation="softmax"),
    ])

# class SimpleDNN(Model):
#     def __init__(self, input_shape: Tuple, num_classes: int, weights: String = None):
#         super().__init__(input_shape, num_classes, weights)

#         self._model = tf.keras.Sequential([
#             tf.keras.layers.Flatten(input_shape=input_shape),
#             tf.keras.layers.Dense(512, activation='relu'),
#             tf.keras.layers.Dense(256, activation='relu'),
#             tf.keras.layers.Dense(128, activation='relu'),
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dense(self.num_classes, activation='softmax')
#         ])

class SimpleDNN(Model):
    def __init__(self, input_shape: Tuple, num_classes: int, weights: String = None):
        super().__init__(input_shape, num_classes, weights)

        self._model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])


class EfficientNetB0(Model):
    def __init__(self, input_shape: Tuple, num_classes: int, weights: String = None):
        super().__init__(input_shape, num_classes, weights)
        self._model = tf.keras.applications.EfficientNetB0(
            self.input_shape, classes=self.num_classes, weights=self.weights)

# class Vgg16(Model):
#     raise Exception("Not Implemented Yet")

class FMCNNModel(Model):
    def __init__(self, input_shape: Tuple, num_classes: int, weights: String = None):
        super().__init__(input_shape, num_classes, weights)
    
        # Kernel initializer
        kernel_initializer = tf.keras.initializers.glorot_uniform(seed=SEED)

        # Architecture
        inputs = tf.keras.layers.Input(shape=input_shape)
        layers = tf.keras.layers.Conv2D(
            32,
            kernel_size=(5, 5),
            strides=(1, 1),
            kernel_initializer=kernel_initializer,
            padding="same",
            activation="relu",
        )(inputs)
        layers = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layers)
        layers = tf.keras.layers.Conv2D(
            64,
            kernel_size=(5, 5),
            strides=(1, 1),
            kernel_initializer=kernel_initializer,
            padding="same",
            activation="relu",
        )(layers)
        layers = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layers)
        layers = tf.keras.layers.Flatten()(layers)
        layers = tf.keras.layers.Dense(
            512, kernel_initializer=kernel_initializer, activation="relu"
        )(layers)

        outputs = tf.keras.layers.Dense(
            num_classes, kernel_initializer=kernel_initializer, activation="softmax"
        )(layers)

        self._model = tf.keras.Model(inputs=inputs, outputs=outputs)


class FedAVGCNN(Model):
    """ Architecture of CNN model used in original FedAVG paper with Cifiar-10 dataset.
    https://doi.org/10.48550/arXiv.1602.05629
     """
    def __init__(self, input_shape: Tuple, num_classes: int, weights: String = None):
        super().__init__(input_shape, num_classes, weights)
        print(input_shape)

        self._model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=64, padding='same', kernel_size=(5,5), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax'),
        ])
