import tensorflow as tf
import tensorflow_federated as tff
import os
import numpy as np
import pickle as pkl
import argparse
from tqdm import tqdm



def create_user_based_data(data):
    # takes a test data or train data of type tensorflow_federated.python.simulation.datasets.client_data.PreprocessClientData and 
    # return user partitioned dataset like "user_id" : [data ...]
    out_json = {}
    for cid in data.client_ids:
        c_data = []
        td = data.create_tf_dataset_for_client(cid)
        for t in td.take(-1):
            c_data.append(t['snippets'].numpy().decode('utf-8'))
        out_json[cid] = c_data
    return out_json

def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def partition_data(seq_length, out_dir):
    """load  and partitions the shakespare dataset."""

    # A fixed vocabularly of ASCII chars that occur in the works of Shakespeare and Dickens:
    vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r}')

    # Creating a mapping from unique characters to indices
    char_to_int = {u:i for i, u in enumerate(vocab)}
    int_to_char = np.array(vocab)

    print(f"Getting Shakespeare dataset from tensorflow federated")
    train_data_raw, test_data_raw = tff.simulation.datasets.shakespeare.load_data()
    train_data = create_user_based_data(train_data_raw)
    test_data = create_user_based_data(test_data_raw)
    all_train_data = []
    all_test_data = []

    for cid in tqdm(train_data.keys(), desc="Processing", unit=" Clients"):
        c_train_data = train_data[cid]
        c_test_data = test_data[cid]

        train_text = " ".join(c_train_data)
        test_text = " ".join(c_test_data)
        if  len(train_text) and len(test_text) >= seq_length: #take only data from clients where whole taken is greater than seq_len characters
            train_sequences = []
            train_next_chars = []
            for i in range(0, len(train_text) - seq_length, 1):
                train_sequences.append(train_text[i:i+seq_length])
                train_next_chars.append(train_text[i+seq_length])
            test_sequences = []
            test_next_chars = []
            for i in range(0, len(test_text) - seq_length, 1):
                test_sequences.append(test_text[i:i+seq_length])
                test_next_chars.append(test_text[i+seq_length])

            X_train = []
            y_train = []
            for i in range(len(train_sequences)):
                X_train.append([char_to_int[char] for char in train_sequences[i]])
                y_train.append(char_to_int[train_next_chars[i]])
            X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=seq_length, padding='pre')
            y_train = tf.keras.utils.to_categorical(y_train,len(char_to_int.keys()))
            X_test = []
            y_test = []
            for i in range(len(test_sequences)):
                X_test.append([char_to_int[char] for char in test_sequences[i]])
                y_test.append(char_to_int[test_next_chars[i]])
            X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=seq_length, padding='pre')
            y_test = tf.keras.utils.to_categorical(y_test,len(char_to_int.keys()))

            if len(X_train) and len(y_train) > 0:
                all_train_data.append((X_train,y_train))
                all_test_data.append((X_test, y_test))

    create_dir_if_not_exists(os.path.join(out_dir,'train'))
    create_dir_if_not_exists(os.path.join(out_dir,'test'))
    with open(os.path.join(out_dir,'train','shakespare_train_processed.pkl'), 'wb') as out_file:
        pkl.dump(all_train_data, out_file)
    with open(os.path.join(out_dir,'test','shakespare_test_processed.pkl'), 'wb') as out_file:
        pkl.dump(all_test_data, out_file)
    print(f"Processed data saved at : {out_dir}")
    print(f"Total clients in preprocessed dataset : {len(all_train_data)} with sequence length : {seq_length}")
    return all_train_data, all_test_data

if __name__ == "__main__":

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="CLI options.")

    # Define command-line arguments
    parser.add_argument('--seq_length', type=int, help='Sequence length of input data', required= True)
    parser.add_argument('--out_dir', type=str, help='Location where pre-processed data would be saved',required=True )

    # Parse the command-line arguments
    args = parser.parse_args()

    partition_data(seq_length=args.seq_length,out_dir=args.out_dir)