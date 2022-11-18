from mak.data.dataset import Dataset
import tensorflow as tf
import numpy as np
from typing import Tuple, cast
import string
SEED = 2000


class FashionMnistData(Dataset):
    def __init__(self, num_clients: int, data_distribution: string = "iid"):
        super().__init__(num_clients, data_distribution)
        self.x_train, self.y_train, self.x_test, self.y_test = self._get_and_preprocess_data()

    def _get_data_stats(self):
        # simply show the stats about data set
        labels = np.argmax(tf.keras.utils.to_categorical(self.y_train, 10))
        all_classes = np.unique(self.all_labels)
        print(
            f"Dataset : FashionMNIST => Num Classes : {len(all_classes)} Class Labels = {all_classes}")
        print(
            f"Data Stats => X Train Samples : {len(self.x_train)}  X Test Samples {len(self.x_test)}")
        print(
            f"Data Stats => Y Train Samples : {len(self.y_train)}  Y Test Samples {len(self.y_test)}")

    def _get_and_preprocess_data(self):
        (x_train, y_train), (x_test,
                             y_test) = tf.keras.datasets.fashion_mnist.load_data()

        # x_train, y_train = shuffle(x_train, y_train)
        # x_test, y_test = shuffle(x_test, y_test)

        self.all_labels = np.array(y_train)

        # Adjust x sets shape for model
        x_train = self.adjust_x_shape(x_train)
        x_test = self.adjust_x_shape(x_test)
        # Normalize data
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        return x_train, y_train, x_test, y_test

    def adjust_x_shape(self, nda: np.ndarray) -> np.ndarray:
        """Turn shape (x, y, z) into (x, y, z, 1)."""
        nda_adjusted = np.reshape(
        nda, (nda.shape[0], nda.shape[1], nda.shape[2], 1))
        return cast(np.ndarray, nda_adjusted)

    # data partitioning methods

    def load_data_iid(self, id: int):
        # divides dataset into equall sized partitions based on number of clients
        # this method takes partition id as argument
        num_samples_train = int(len(self.x_train) / self.num_clients)
        num_samples_valid = int(len(self.x_test) / self.num_clients)

         # Convert class vectors to one-hot encoded labels
        y_train = tf.keras.utils.to_categorical(self.y_train, 10)
        y_test = tf.keras.utils.to_categorical(self.y_test, 10)

        return (
        self.x_train[id * num_samples_train: (id + 1) * num_samples_train],
        y_train[id * num_samples_train: (id + 1) * num_samples_train],
    ), (
        self.x_test[id * num_samples_valid: (id + 1) * num_samples_valid],
        y_test[id * num_samples_valid: (id + 1) * num_samples_valid],
    )

    def load_data_iid_custom(self, id: int, num_train_samples: int, num_valid_samples: int) -> Tuple:
        # provides a data partition based on provided num_samples
         # Convert class vectors to one-hot encoded labels
        y_train = tf.keras.utils.to_categorical(self.y_train, 10)
        y_test = tf.keras.utils.to_categorical(self.y_test, 10)
        return (
        self.x_train[id * num_train_samples: (id + 1) * num_train_samples],
        y_train[id * num_train_samples: (id + 1) * num_train_samples],
    ), (
        self.x_test[id * num_valid_samples: (id + 1) * num_valid_samples],
        y_test[id * num_valid_samples: (id + 1) * num_valid_samples],
    )

    def load_data_one_class(self, class_id: int):
        assert class_id in range(10)
        # returns all the samples of given class
        class_type_int = int(class_id)

        x_train = self.x_train[self.y_train == class_type_int]
        y_train = self.y_train[self.y_train == class_type_int]

        x_test = self.x_test[self.y_test == class_type_int]
        y_test = self.y_test[self.y_test == class_type_int]

        # Convert class vectors to one-hot encoded labels
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        return (x_train, y_train), (x_test, y_test)

    def load_test_data(self):
    # returns all the test data samples
     # Convert class vectors to one-hot encoded labels
        y_test = tf.keras.utils.to_categorical(self.y_test, 10)
        return (self.x_test, y_test)

    def load_data_two_classes(self, class_1: int, class_2: int):
        assert class_1 in range(10)
    # two class non_iid data
        x_train_1 = self.x_train[self.y_train == class_1]
        y_train_1 = self.y_train[self.y_train == class_1]
    # Take only 3000 samples of the first class
        x_train_1 = x_train_1[:3000]
        y_train_1 = y_train_1[:3000]

        x_test_1 = self.x_test[self.y_test == class_1]
        y_test_1 = self.y_test[self.y_test == class_1]

        x_test_1 = x_test_1[:500]
        y_test_1 = y_test_1[:500]

        x_train_2 = self.x_train[self.y_train == class_2]
        y_train_2 = self.y_train[self.y_train == class_2]

        x_train_2 = x_train_2[:3000]
        y_train_2 = y_train_2[:3000]

        x_test_2 = self.x_test[self.y_test == class_2]
        y_test_2 = self.y_test[self.y_test == class_2]

        x_test_2 = x_test_2[:500]
        y_test_2 = y_test_2[:500]

    # Combine these 2 part of samples
        x_train = np.concatenate((x_train_1, x_train_2))
        y_train = np.concatenate((y_train_1, y_train_2))

        x_test = np.concatenate((x_test_1, x_test_2))
        y_test = np.concatenate((y_test_1, y_test_2))
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        return (x_train, y_train), (x_test, y_test)

    def load_data_ten_classes(self, train_dist, test_dist):
    # generates 10 class data accoding to given distributuion
        dx_train = []
        dy_train = []
        dx_test = []
        dy_test = []
        counts = [0 for i in range(10)]
        for i in range(len(self.x_train)):
            if counts[self.y_train[i]] < train_dist[self.y_train[i]]:
                dx_train.append(self.x_train[i])
                dy_train.append(self.y_train[i])
                counts[self.y_train[i]] += 1
        counts = [0 for i in range(10)]

        for i in range(len(self.x_test)):
            if counts[self.y_test[i]] < test_dist[self.y_test[i]]:
                dx_test.append(self.x_test[i])
                dy_test.append(self.y_test[i])
                counts[self.y_test[i]] += 1

        # # Convert class vectors to one-hot encoded labels
        dy_train = tf.keras.utils.to_categorical(dy_train, 10)
        dy_test = tf.keras.utils.to_categorical(dy_test, 10)

        return (np.array(dx_train), np.array(dy_train)), (np.array(dx_test), np.array(dy_test))

    def load_data_majority_class(self, class_id, percent):
        class_type_int = int(class_id)
        majority_class_per = int(percent * 100)
        minority_classes_per = int(100-majority_class_per)
        total_train_samples = 5000
        total_test_samples = 1000

        num_samples_majority_train = int(
            (total_train_samples/100)*majority_class_per)
        num_samples_majority_test = int(
            (total_test_samples/100)*majority_class_per)

        num_minority_samples_train = int(
            (total_train_samples - num_samples_majority_train) / 9)
        num_minority_samples_test = int(
            (total_test_samples - num_samples_majority_test) / 9)

        train_dist = []
        test_dist = []

        for i in range(0, 10):
            if i == class_type_int:
                train_dist.append(num_samples_majority_train)
                test_dist.append(num_samples_majority_test)
            else:
                train_dist.append(num_minority_samples_train)
                test_dist.append(num_minority_samples_test)

        print(train_dist)
        print(test_dist)

        dx_train = []
        dy_train = []
        dx_test = []
        dy_test = []
        counts = [0 for i in range(10)]
        for i in range(len(self.x_train)):
            if counts[self.y_train[i]] < train_dist[self.y_train[i]]:
                dx_train.append(self.x_train[i])
                dy_train.append(self.y_train[i])
                counts[self.y_train[i]] += 1
        counts = [0 for i in range(10)]

        for i in range(len(self.x_test)):
            if counts[self.y_test[i]] < test_dist[self.y_test[i]]:
                dx_test.append(self.x_test[i])
                dy_test.append(self.y_test[i])
                counts[self.y_test[i]] += 1

        # # Convert class vectors to one-hot encoded labels
        dy_train = tf.keras.utils.to_categorical(dy_train, 10)
        dy_test = tf.keras.utils.to_categorical(dy_test, 10)

        return (np.array(dx_train), np.array(dy_train)), (np.array(dx_test), np.array(dy_test))


def shuffle(x_orig: np.ndarray, y_orig: np.ndarray, seed: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle x and y in the same way."""
    np.random.seed(seed)
    idx = np.random.permutation(len(x_orig))
    # print(idx[:20])
    return x_orig[idx], y_orig[idx]