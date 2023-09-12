from mak.data.dataset import Dataset
import tensorflow as tf
import numpy as np
from typing import Tuple, cast
import string
import random
import pickle as pkl
SEED = 2000


class ShakespeareData(Dataset):
     """Shakespeare dataset for next character prediction.
   
     Args:
        num_clients: int number of clients you wish to train the model on
        train_file: preprocessed pickle file containing the training data
        test_file : preprocessed pickle file containing the testing data
        data_distribution : string ignore for now as shakespeare is one class non-iid.

    """
     def __init__(self, num_clients: int, train_file, test_file, data_distribution: string = "iid"):
        super().__init__(num_clients, data_distribution)
        self.train_file = train_file
        self.test_file = test_file
        self.train_data, self.test_data = self._get_and_preprocess_data()

     def _get_and_preprocess_data(self):
        """
        Returns : client partitoned train and test datasets
        """
        with open(self.train_file,'rb') as f:
            train_data = pkl.load(f)
        with open(self.test_file,'rb') as f:
            test_data = pkl.load(f)
    
        return train_data, test_data
     
     def get_client_data(self,cid:int):
        """
        Args:
            cid : int id of client for which data is needed.

        Returns: A preprocessed Tuple of train and test data set for the current client
        """
        return self.train_data[cid], self.test_data[cid]
     
     def load_test_data(self):
        cid = random.randint(0,len(self.test_data)-1)
        data = self.test_data[cid]
        while len(data[0]) == 0:
            cid = random.randint(0,len(self.test_data)-1)
            data = self.test_data[cid]
        return data
     

  