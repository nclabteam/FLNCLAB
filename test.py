from mak.data.fashion_mnist import FashionMnistData
from mak.model.models import SimpleCNN
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
import random
import pandas as pd
import tensorflow as tf
from typing import cast
def adjust_x_shape(nda: np.ndarray) -> np.ndarray:
        """Turn shape (x, y, z) into (x, y, z, 1)."""
        nda_adjusted = np.reshape(
        nda, (nda.shape[0], nda.shape[1], nda.shape[2], 1))
        return cast(np.ndarray, nda_adjusted)
input_shape = (28, 28, 1)
model = SimpleCNN(input_shape=input_shape, num_classes=10)._model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Adjust x sets shape for model
x_train = adjust_x_shape(x_train)
x_test = adjust_x_shape(x_test)
# # Normalize data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_test = tf.keras.utils.to_categorical(y_test, 10)
y_train = tf.keras.utils.to_categorical(y_train, 10)

print(x_train.shape)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=["accuracy"],
    run_eagerly=True,
)

save = tf.keras.callbacks.ModelCheckpoint(
    './saved_model1',
    monitor="val_loss",
    verbose=1,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    options=None,
    initial_value_threshold=None,
)

hist = model.fit(x=x_train,y=y_train,epochs=100,validation_split = 0.2,callbacks = [save])
history = hist.history
df =  pd.DataFrame(history)
df.to_csv('history.csv')

load_model = tf.keras.models.load_model('./saved_model1')
loss, accuracy = load_model.evaluate(x_test, y_test, verbose=1)
print(f"Loss : {loss}  Accuracy : {accuracy}")


# def plot_data(x_train, y_train):
#     num_row = 2
#     num_col = 5
#     fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col, 2*num_row))
#     for i in range(10):
#         idx = random.randint(0,len(x_train))
#         ax = axes[i//num_col, i % num_col]
#         ax.imshow(x_train[idx], cmap='gray')
#         # ax.set_title('{}'.format(class_names[y_train[i]]))
#         ax.set_title(f"{class_names[np.argmax(y_train[idx])]}")
#     plt.tight_layout()
#     plt.show()

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# train_dist = [500,500,500,500,500,500,500,500,500,500]
# test_dist = [100,100,100,100,100,100,100,100,100,100]
# dataset = FashionMnistData(10)
# (x_train, y_train), (x_test, y_test) = dataset.load_data_ten_classes(train_dist=train_dist,test_dist=test_dist)
# # (x_train, y_train) = dataset.load_test_data()
# print(y_train.shape)
# plot_data(x_train,y_train)
# # print(dataset._get_data_stats())

# class Vehicle():
#     def __init__(self,type,gears=3) -> None:
#         self.type = type
#         self.gears = gears

#     def get_vehicle_type(self):
#         print(f"Vehicle Type = {self.type}")

#     def get_num_gears(self):
#         print(f"Gears = {self.gears}")

# class Bus(Vehicle):
#     def __init__(self, type, gears,capacity) -> None:
#         super().__init__(type, gears)
#         self.capacity = capacity

#     def get_capacity(self):
#         print(f"Capacity = {self.capacity}")

# class MiniBus(Bus):
#     def __init__(self, type, gears, capacity,size) -> None:
#         super().__init__(type, gears, capacity)
#         self.size = size

#     def get_size(self):
#         print(f"Size = {self.size}")

# mb = MiniBus("mini",2,22,220)

# mb.get_vehicle_type()
# mb.get_num_gears()
# mb.get_capacity()
# mb.get_size()

# yaml test
# import yaml
# with open('config.yaml') as file:
#     try:
#         config = yaml.safe_load(file)
#         print(config['common']['data_type'])
#     except yaml.YAMLError as exc:
#         print(exc)
