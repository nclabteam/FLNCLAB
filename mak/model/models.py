import enum
from tokenize import String
from typing import Tuple
import tensorflow as tf

BN_AXIS = 3

class Model:
    def __init__(self, input_shape: Tuple, num_classes: int, weights: String = None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = weights
        self._model = None

    def model_details(self):
        print("input size : {}  classes : {} weights : {} ".format(
            self.input_shape, self.num_classes, self.weights))

class LSTMModel(Model):
     """Create a LSTM model for next character task.
    Args:
        input_shape:  sequence length of your data.
        num_classes : int the size of the vocabulary of dataset

    Returns:
        A keras model.

    """
     def __init__(self, input_shape: Tuple, num_classes: int, weights: String = None):
        super().__init__(input_shape, num_classes, weights)
        self.input_shape = input_shape
        self.input_length = self.input_shape
        self._model = tf.keras.Sequential([
        tf.keras.layers.Embedding(num_classes, 256, input_length=self.input_length),
        tf.keras.layers.LSTM(256),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])


class MobileNetV2(Model):
    def __init__(self, input_shape: Tuple, num_classes: int, weights: String = None):
        super().__init__(input_shape, num_classes, weights)
        base_model = tf.keras.applications.MobileNetV2(
        input_shape=self.input_shape,
        include_top=False,
        weights=self.weights
    )
        # Freeze the pre-trained model weights
        base_model.trainable = True

        for layer in base_model.layers[:100]:
            layer.trainable =  False
        # Trainable classification head
        maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
        prediction_layer = tf.keras.layers.Dense(units=self.num_classes, activation='softmax')
        # Layer classification head with feature detector
        self._model = tf.keras.models.Sequential([
            base_model,
            maxpool_layer,
            prediction_layer
        ])

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


class FMCNNModel(Model):
    def __init__(self, input_shape: Tuple, num_classes: int, weights: String = None):
        super().__init__(input_shape, num_classes, weights)
    
        # Kernel initializer
        kernel_initializer = tf.keras.initializers.glorot_uniform(seed=123)

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
        
        self._model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=64, padding='same', kernel_size=(5,5), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax'),
        ])


class ResNet18(Model):
    """Implementation of ResNet-18 architecture.
    This is not tested needs to be tested
    """
    def __init__(self, input_shape, num_classes, weights=None,layer_params=[2, 2, 2, 2], pooling='avg'):
        super().__init__(input_shape, num_classes, weights)
        self.layer_params = layer_params
        self.pooling = pooling
        img_input = tf.keras.layers.Input(shape=input_shape)


        x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
        x = tf.keras.layers.Conv2D(64, (7, 7),
                        strides=(2, 2),
                        padding='valid',
                        kernel_initializer='he_normal',
                        name='conv1')(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv1')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.make_basic_block_layer(x, filter_num=64,
                                    blocks=self.layer_params[0])
        x = self.make_basic_block_layer(x, filter_num=128,
                                            blocks=self.layer_params[1],
                                            stride=2)
        x = self.make_basic_block_layer(x, filter_num=256,
                                            blocks=self.layer_params[2],
                                            stride=2)
        x = self.make_basic_block_layer(x, filter_num=512,
                                            blocks=self.layer_params[3],
                                            stride=2)

        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        self._model = tf.keras.Model(img_input, outputs, name='ResNet-18')

            
    def make_basic_block_base(self,inputs, filter_num, stride=1):
        x = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            kernel_initializer='he_normal',
                                            padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization(axis=BN_AXIS)(x)
        x = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            kernel_initializer='he_normal',
                                            padding="same")(x)
        x = tf.keras.layers.BatchNormalization(axis=BN_AXIS)(x)

        shortcut = inputs
        if stride != 1:
            shortcut = tf.keras.layers.Conv2D(filters=filter_num,
                                                kernel_size=(1, 1),
                                                strides=stride,
                                                kernel_initializer='he_normal')(inputs)
            shortcut = tf.keras.layers.BatchNormalization(axis=BN_AXIS)(shortcut)

        x = tf.keras.layers.add([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)

        return x

    def make_basic_block_layer(self,inputs, filter_num, blocks, stride=1):
        x = self.make_basic_block_base(inputs, filter_num, stride=stride)

        for _ in range(1, blocks):
            x = self.make_basic_block_base(x, filter_num, stride=1)

        return x
   

    