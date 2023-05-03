This file describes the models already implemented in this framework.

The code of all these model classes can be found inside [`mak/model/models.py`](../mak/model/models.py) file.
## Models
1. `mobilenetv2:` MobileNet V2 is a Keras based implementation of https://arxiv.org/abs/1801.04381) (CVPR 2018) paper. The Detailed information regarding this architecture can be obtained from keras official documentation [here](https://keras.io/api/applications/mobilenet/#mobilenetv2-function).
    
2. `simplecnn:` It is a simple CNN model implemented from scratch whose architecture is shown below:
    ```
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
    ```
3. `simplednn:` It is a simple Deep neural network model implemented from scratch whose architecture is shown below:
    ```
    class SimpleDNN(Model):
    def __init__(self, input_shape: Tuple, num_classes: int, weights: String = None):
        super().__init__(input_shape, num_classes, weights)

        self._model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
    ```
4. `kerasexpcnn:` It is also a simple CNN architecture used for image classification, actually this architecture is inspired from a tutorial from Keras documentation. The tutorial can be found [here](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
    ```
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
    ```
5. `mnistcnn:` It is also a simple CNN model implemented from scratch whose architecture is shown below:
    ```
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
    ```
6. `efficientnet:` This is also a Keras based implementation of `efficientnetB0` (https://arxiv.org/abs/1905.11946 (ICML 2019)) paper. The Detailed information regarding this architecture can be obtained from keras official documentation [here](https://keras.io/api/applications/efficientnet/).
7. `fedavgcnn:` It is also a simple CNN model implemented from scratch whose used in original FedAVG ( https://doi.org/10.48550/arXiv.1602.05629) paper and the architecture is shown below:
    ```
     self._model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=64, padding='same', kernel_size=(5,5), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax'),
        ])
    ```
8. `fmcnn:` A simple CNN model implemented from scratch, the architecture is shown below:
    ```
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
    ```

