'''
Author: Caleb Moses
Date: 04-06-2017

This file trains a character-level multi-layer RNN on text data.

Code is based on Andrej Karpathy's implementation in Torch at:
https://github.com/karpathy/char-rnn/blob/master/train.lua

I modified the model to run using TensorFlow and Keras. Supports GPUs, 
as well as many other common model/optimization bells and whistles.

TO DO: 
- Add learning rate
- Improve tensorboard logs
- Implement optional text sample log callback
- Add other optimizers?
- Fix not-displaying validation loss
- Optionally suppress warnings?
- Tidy up print statements
- Implement recursive file search

DONE:
- Double check generators are correct (i.e. correctly assign training/
  validation data in the correct proportions)
- Implement shuffling of data for training/validation set
'''

import os, re, random
import sys, argparse, codecs
import itertools as it
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import np_utils


def parse_args():
    '''Parses all keyword arguments for model and returns them.

       Returns:
        - data_dir:   (str) The directory to the text file(s) for training.
        - rnn_size:   (int) The number of cells in each hidden layer in 
                      the network.
        - num_layers: (int) The number of hidden layers in the network.
        - dropout:    (float) Dropout value (between 0, 1 exclusive).'''

    # initialise parser
    parser = argparse.ArgumentParser()

    # add arguments, set default values and expected types
    parser.add_argument("-data_dir",
        help="The directory to the text file(s) for training.")
    parser.add_argument("-seq_length", type=int, default=25,
        help="The length of sequences to be used for training")
    parser.add_argument("-validation_split", type=float, default=0.1,
        help="The proportion of the training data to use for validation")
    parser.add_argument("-batch_size", type=int, default=100,
        help="The number of minibatches to be used for training")
    parser.add_argument("-rnn_size", type=int, default=128,
        help="The number of cells in each hidden layer in the network")
    parser.add_argument("-num_layers", type=int, default=3,
        help="The number of hidden layers in the network")
    parser.add_argument("-dropout", type=float, default=0.1,
        help="Dropout value (between 0, 1 exclusive)")
    parser.add_argument("-epochs", type=int, default=20,
        help="Number of epochs for training")
    parser.add_argument("-verbose", type=int, default=1,
        help="Number of epochs for training")
    parser.add_argument("-tensorboard", type=int, default=1,
        help="Save model statistics to tensorboard")
    
    # parse arguments and return their values
    args = parser.parse_args()
    return args.data_dir, args.seq_length, args.validation_split, \
           args.batch_size, args.rnn_size, args.num_layers, args.dropout, \
           args.epochs, args.verbose, args.tensorboard


def print_data(text):
    '''Re-encodes text so that it can be printed to command line 
       without raising a UnicodeEncodeError, and then prints it.
       Incompatible characters are simply dropped before printing.

       Args:
       - text: (str) The text to be printed'''

    print(text.encode(sys.stdout.encoding, errors='replace'))


def load_data(data_dir, encoding='utf-8'):
    '''Appends all text files in data_dir into a single string and returns it.
       All files are assumed to be utf-8 encoded, and of type '.txt'.

       Args:
       - data_dir: (str) The directory to text files for training.
       - encoding: (str) The type of encoding to use when decoding each file.

       Returns:
       - text_data: (str) Appended files as a single string.'''

    print("Loading data from %s" % os.path.abspath(data_dir))
    # Initialise text string
    text_data = ''
    # select .txt files from data_dir
    for filename in filter(lambda s: s.endswith(".txt"), os.listdir(data_dir)):
        # open file with default encoding
        print("loading file: %s" % filename)
        filepath = os.path.abspath(os.path.join(data_dir, filename))
        with open(filepath,'r', encoding = encoding) as f:
            text_data += f.read() + "\n"
    return text_data


def process_text(text_data, seq_length):
    '''Preprocesses text_data for RNN model.

       Args:
       - text: (str) text file to be processed.
       - seq_length: (int) length of character sequences to be considered 
                     in the training set.

       Returns:
       - char_to_int: (dict) Maps characters in the character set to ints.
       - int_to_char: (dict) Maps ints to characters in the character set.
       - n_chars: (int) The number of characters in the text.
       - n_vocab: (int) The number of unique characters in the text.'''

    # create mapping of unique chars to integers, and a reverse mapping
    chars = sorted(set(text_data))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}

    # summarize the loaded data
    n_chars = len(text_data)
    n_vocab = len(chars)
    
    return char_to_int, int_to_char, n_chars, n_vocab


def get_batch(batch, starts, text_data, seq_length, batch_size, 
              char_to_int, n_vocab):
    '''A generator that returns sequences of length seq_length, in
       batches of size batch_size.
       
       Args:
       - batch: (int) The index of the batch to be returned
       - text_data: (str) The text to feed the model
       - seq_length: (int) The length of each training sequence
       - batch_size: (int) The size of minibatches for training'''
    
    # prepare the dataset of input to output pairs encoded as integers
    dataX = []
    dataY = []
    for start in range(batch_size * batch, batch_size * (batch + 1)): 
        seq_in  = text_data[starts[start]:starts[start] + seq_length]
        seq_out = text_data[starts[start] + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
        
    X = np_utils.to_categorical(dataX, num_classes=n_vocab)
    X = X.reshape(batch_size, seq_length, n_vocab)

    # one hot encode the output variable
    y = np_utils.to_categorical(dataY, num_classes=n_vocab)
    
    return X, y


def generate_batches(mode, text_data, seq_length, validation_split,
                     batch_size, char_to_int, n_chars, n_vocab,
                     random_seed=1234, shuffle=True):
    '''A generator that returns training sequences of length seq_length, in
       batches of size batch_size.

       Args:
       - mode: (str) Whether the batch is for training or validation. 
               'validation' or 'train' only
       - text_data: (str) The text for training
       - seq_length: (int) The length of each training sequence
       - batch_size: (int) The size of minibatches for training
       - validation_split: (float) The proportion of batches to use as 
                           validation data
       - random_seed: A random seed'''

    # set random seed
    random.seed(random_seed)
    
    # index the text_data
    starts = list(range(n_chars - n_chars % seq_length - seq_length))
    
    if shuffle:
        # shuffle the indices
        random.shuffle(starts)
    
    while True:
        n_batches = n_chars // batch_size
        validation_size = round(n_batches * validation_split)
        if mode == 'validation':
            for batch in range(validation_size):
                X, y = get_batch(batch, starts, text_data, seq_length, 
                                 batch_size, char_to_int, n_vocab)
                yield X, y
                
        elif mode == 'train':
            for batch in range(validation_size, n_batches):
                X, y = get_batch(batch, starts, text_data, seq_length, 
                                 batch_size, char_to_int, n_vocab)
                yield X, y
        else:
            raise ValueError("only 'validation' and 'train' modes accepted")


def build_model(batch_size, seq_length, n_vocab, 
                rnn_size, num_layers, drop_prob):
    '''Defines the RNN LSTM model.

       Args:
        - batch_size: (int) The size of each minibatches.
        - seq_length: (int) The length of each sequence for the model.
        - rnn_size: (int) The number of cells in each hidden layer.
        - num_layers: (int) The number of hidden layers in the network.
        - drop_prob: (float) The proportion of cells to drop in each dropout 
                             layer.
       Returns:
        - model: (keras.models.Sequential) The constructed Keras model.'''

    model = Sequential()
    for i in range(num_layers):
        if i == num_layers - 1:
            # add last hidden layer
            model.add(LSTM(rnn_size, return_sequences=False))
        elif i == 0:
            # add first hidden layer - This crashes if num_layers == 1
            model.add(LSTM(rnn_size, 
                           batch_input_shape=(None, seq_length, n_vocab),
                           return_sequences=True))
        else:
            # add middle hidden layer
            model.add(LSTM(rnn_size, return_sequences=True))
        model.add(Dropout(drop_prob))
    # add output layer
    model.add(Dense(n_vocab, activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])  

    return model


def set_callbacks(verbose, use_tensorboard, checkpoint_dir = "checkpoints"):
    '''Set callbacks for Keras model.

       Args:
         - use_tensorboard: (int) Add TensorBoard callback if use_tensorboard == 1

       Returns:
         - callbacks: (list) list of callbacks for model'''        
    root_dir = '..'
    checkpoint_dir = os.path.join(root_dir,
                                  checkpoint_dir, 
                                  'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    callbacks = [ModelCheckpoint(checkpoint_dir, verbose=verbose)]
    if use_tensorboard:
        log_dir = os.path.join('..', 'logs')
        tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=0.01,
                              write_images=True)
        callbacks.append(tb_callback)  

    return callbacks


def fit_model(model, text_data, seq_length, validation_split, epochs, 
              batch_size, char_to_int, n_chars, n_vocab, verbose, use_tensorboard):
    '''Trains the model on the training data.

       Args:
       - model:
       - text_data:
       - seq_length:
       - batch_size:
       - char_to_int:'''
    n_batches = len(text_data) // batch_size
    batch_params = (text_data, seq_length, validation_split,
                     batch_size, char_to_int, n_chars, n_vocab)
    hist = model.fit_generator(
               generator = generate_batches('train', *batch_params),
               validation_data = generate_batches('validation', *batch_params),
               validation_steps = int(n_batches * validation_split),
               workers = 1,
               epochs = epochs,
               steps_per_epoch = n_batches,
               verbose = verbose,
               callbacks = set_callbacks(verbose, use_tensorboard))
    return hist


def Main():
    '''Executes the model'''

    # load text data to memory
    text_data = load_data(data_dir)

    # preprocess the text - construct character dictionaries etc
    char_to_int, int_to_char, n_chars, n_vocab = \
                                process_text(text_data, seq_length)

    # build and compile Keras model
    model = build_model(batch_size, seq_length, n_vocab,
                        rnn_size, num_layers, drop_prob)

    # fit model using generator
    hist = fit_model(model, text_data, seq_length, validation_split, epochs,
                     batch_size, char_to_int, n_chars, n_vocab,  
                     verbose, use_tensorboard)


if __name__ == "__main__":

    # parse keyword arguments
    data_dir, seq_length, validation_split, batch_size, rnn_size, \
    num_layers, drop_prob, epochs, verbose, use_tensorboard = parse_args()

    Main()
