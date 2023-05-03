This file describes the different data distribution avalaible and how the data is splitted based on these data distributions.

## Data Distributions:
1. `iid` : If the data distribution used is `iid` the data will be equally partitioned among all clients i.e every client will recieve data from all the classes of the selected dataset.
2. `one-class-niid` : If the data distribution used is `one-class-niid` the data will be partitioned among all clients based on class labels. i.e every client will recieve data samples from only one class of data.
3. `one-class-niid-majority` : It implies the majority(75% by default) of the data recieved by a client will come from single class and rest of the data (25%) will be equally taken from remaining 09 classes.
4. `two-class-niid` : This distribution is the most common non-iid data distribution in federated learning literature. This is used by many classical papers like `FedAVG`. In this data is distributed among clients such as each client recieved data from only two classes from the dataset.
5. `dirichlet_niid` : This is a relatively new approach to generate non-iid data based on the dirichlet distribution process. Some more details regarding this process can be obtained from here https://arxiv.org/abs/1909.06335 .