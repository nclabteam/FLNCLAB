from abc import abstractmethod
import string

class Dataset:
    def __init__(self, num_clients: int, data_distribution: string = "iid"):
        self.num_clients = num_clients
        self.data_distribution = data_distribution

    @abstractmethod
    def _get_and_preprocess_data(self):
        pass