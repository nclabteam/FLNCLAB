from mak.data.dataset import Dataset
import tensorflow as tf
import numpy as np
from typing import Tuple, cast
import string
import random
import os
import cv2
from sklearn.model_selection import train_test_split
SEED = 2000


class ViolationDetection(Dataset):
    def __init__(self, num_clients: int, data_root: string,image_size:tuple, data_distribution: string = "iid"):
        super().__init__(num_clients, data_distribution)
        self.class_ids = {'violate':0, 'non_violate' : 1}
        self.image_size = image_size
        self.data_root = data_root

        self.client_ids = sorted(os.listdir(self.data_root))
        self.client_stats = {}
        self.classes = os.listdir(os.path.join(self.data_root,self.client_ids[0]))
        for client_id in self.client_ids:
            d = { }
            for class_name in self.classes:
                d[class_name] = len(os.listdir(os.path.join(self.data_root,client_id,class_name)))
            self.client_stats[client_id] = d

    def _get_data_stats(self):
        # simply show the stats about data set
        non_violate_sum = 0
        violate_sum = 0

        # Iterate through the dictionary to calculate the sums
        for client_data in self.client_stats.values():
            non_violate_sum += client_data['non_violate']
            violate_sum += client_data['violate']
        print(
            f"Dataset : Violation Detection Dataset => Num Clients : {len(self.client_ids)} Num Classes : {len(self.classes)} Class Labels = {self.classes}")
        print(f"Total Samples : {violate_sum+non_violate_sum}, Violate Samples : {violate_sum} Non-Violate Samples : {non_violate_sum}")
        print("Client Wise data Distribution")
        for c in self.client_stats.keys():
            print(f"Client : {c} Stats: {self.client_stats[c]}")
        

    def get_client_data(self,cid):
        client_dir = os.path.join(self.data_root, cid)
        # print(f"reading data for client : {cid} path : {client_dir}")
        if not os.path.exists(client_dir):
            print(f"Cannot find the data for client {cid} at loc: {client_dir}")
            return None, None

        data = []
        labels = []
        class_directories = [d for d in os.listdir(client_dir) if os.path.isdir(os.path.join(client_dir, d))]
        
        for class_dir in class_directories:
            class_path = os.path.join(client_dir, class_dir)
            class_label = class_dir

            for filename in os.listdir(class_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(class_path, filename)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, self.image_size)
                    img = img / 255.0  # Normalize pixel values to [0, 1]
                    data.append(img)
                    labels.append(self.class_ids[class_label])

        data = np.array(data)
        labels = np.array(labels)
        x_train, x_test, y_train, y_test = train_test_split(data,labels,test_size=0.2)
        x_train, y_train = shuffle(x_train, y_train)
        x_test, y_test = shuffle(x_test, y_test)
        y_train = tf.keras.utils.to_categorical(y_train, len(self.classes))
        y_test = tf.keras.utils.to_categorical(y_test, len(self.classes))
        return (x_train,y_train),(x_test,y_test)
                

    def load_test_data(self):
     # Convert class vectors to one-hot encoded labels
        (x_train,y_train),(x_test,y_test) = self.get_client_data(random.choice(list(self.client_ids)))
        return (x_test, y_test)
    
    def load_all_data(self):
        """Load all the avalaible data from all the clients as one data set"""
       # Initialize empty arrays for the combined data
        feature_dim = self.image_size[0]*self.image_size[1]*3
        x_train_all = []
        y_train_all = []
        x_test_all = []
        y_test_all = []
        for client in self.client_ids:
            (x_train,y_train),(x_test,y_test) = self.get_client_data(client)
           # Append the data from each folder to the respective lists
            x_train_all.append(x_train)
            y_train_all.append(y_train)
            x_test_all.append(x_test)
            y_test_all.append(y_test)
        # Concatenate the data from all folders
        combined_x_train = np.concatenate(x_train_all, axis=0)
        combined_y_train = np.concatenate(y_train_all, axis=0)
        combined_x_test = np.concatenate(x_test_all, axis=0)
        combined_y_test = np.concatenate(y_test_all, axis=0)
        # combined_x_train, combined_y_train = shuffle(combined_x_train, combined_y_train)
        # combined_x_test,combined_y_test = shuffle(combined_x_test,combined_y_test)
        return (combined_x_train, combined_y_train), (combined_x_test,combined_y_test)


def shuffle(x_orig: np.ndarray, y_orig: np.ndarray, seed: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle x and y in the same way."""
    np.random.seed(seed)
    idx = np.random.permutation(len(x_orig))
    # print(idx[:20])
    return x_orig[idx], y_orig[idx]


  