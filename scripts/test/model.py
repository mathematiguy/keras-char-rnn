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
- Improve TensorBoard logs
'''

import os, re, random
import sys, argparse, codecs
import itertools as it
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.callbacks import TensorBoard


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
    parser.add_argument("-batch_size", type=int, default=100,
        help="The number of minibatches to be used for training")
    parser.add_argument("-rnn_size", type=int, default=128,
        help="The number of cells in each hidden layer in the network")
    parser.add_argument("-num_layers", type=int, default=2,
        help="The number of hidden layers in the network")
    parser.add_argument("-dropout", type=float, default=0.1,
        help="Dropout value (between 0, 1 exclusive)")
    parser.add_argument("-epochs", type=int, default=1,
        help="Number of epochs for training")
    parser.add_argument("-tensorboard", type=int, default=1,
        help="Save model statistics to TensorBoard")

    # parse arguments and return their values
    args = parser.parse_args()
    return args.data_dir, args.seq_length, args.batch_size, args.rnn_size, \
           args.num_layers, args.dropout, args.epochs, args.tensorboard


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
        print("Loading file: %s" % filename)
        filepath = os.path.abspath(os.path.join(data_dir, filename))
        with open(filepath,'r', encoding = encoding) as f:
            text_data += f.read() + "\n"
    return text_data


def get_text_data(text_data):
	# create mapping of unique chars to integers, and a reverse mapping
    chars = sorted(set(text_data))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}

    # summarize the loaded data
    n_text = len(text_data)
    n_chars = len(chars)

    print("n_text:", n_text)
    print("n_chars:", n_chars)

    return char_to_int, n_text, n_chars


def pre_processing(text_data, seq_length, char_to_int, n_text, n_chars):
    '''Preprocesses text_data for RNN model.

       Args:
       - text: (str) text file to be processed.
       - seq_length: (int) length of character sequences to be considered 
                     in the training set.

       Returns:
       - char_to_int: (dict) Maps characters in the character set to ints.
       - int_to_char: (dict) Maps ints to characters in the character set.
       - n_text: (int) The number of characters in the text.
       - n_chars: (int) The number of unique characters in the text.'''
    
    # prepare the dataset of input to output pairs encoded as integers
    dataX = []
    dataY = []
    for start in range(n_text - seq_length):        
        seq_in  = text_data[start:start + seq_length]
        seq_out = text_data[start + seq_length]

        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    X = np.reshape(dataX, (n_text - seq_length, seq_length, 1))

    # normalise X to [0, 1]
    X = X / n_chars

    # one hot encode the output variable
    y = np_utils.to_categorical(dataY, num_classes=n_chars)
    
    return X, y


def build_model(seq_length, n_text, n_chars, rnn_size, num_layers, drop_prob):
    '''Defines the RNN LSTM model.

       Args:
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
            # add first hidden layer
            model.add(LSTM(rnn_size, 
                           input_shape=(seq_length, 1),
                           return_sequences=True))
        else:
            # add middle hidden layer
            model.add(LSTM(rnn_size, return_sequences=True))
        model.add(Dropout(drop_prob))

    # add output layer
    model.add(Dense(n_chars, activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metric=['accuracy', 'loss', 'val_loss'])  	

    return model


def set_callbacks(tensorboard):
	'''Set callbacks for Keras model.

	Args:
	 - tensorboard: (int) Add tensorboard callback if tensorboard == 1

	Returns:
	 - callbacks: (list) list of callbacks for model'''

	callbacks = [ModelCheckpoint(
	    			'checkpoints\\weights.{epoch:02d}-{val_loss:.2f}.hdf5')]
	if tensorboard:
		tb_callback = TensorBoard(log_dir=r'..\logs', histogram_freq=0.01,
								  write_grads=True, write_images=True)
		callbacks.append(tb_callback)  

	return callbacks


# def fit_model(model, X, y, text_data, seq_length, batch_size, char_to_int, 
# 			  n_text, n_chars):
# 	'''Trains the model on the training data.

# 	   Args:
# 	   - model:
# 	   - text_data:
# 	   - seq_length:
# 	   - batch_size:
# 	   - char_to_int:'''

# 	model.fit(X, y, validation_split = 0.3)
# 	return model


def Main():
	# load text data to memory
	text_data = load_data(data_dir)

	# comment
	char_to_int, n_text, n_chars = get_text_data(text_data)

	# preprocess the text - construct character dictionaries etc
	X, y = pre_processing(text_data, seq_length, char_to_int, n_text, n_chars)

	# build and compile Keras model
	model = build_model(seq_length, n_text, n_chars,
	                    rnn_size, num_layers, drop_prob)

	model.fit(X, y, validation_split = 0.3, verbose=2)

	# # fit model using generator
	# model = fit_model(model, X, y, text_data, seq_length, batch_size, 
	# 				  char_to_int, n_text, n_chars)


if __name__ == "__main__":
	# parse keyword arguments
    data_dir, seq_length, batch_size, rnn_size, \
    num_layers, drop_prob, epochs, tensorboard = parse_args()

    Main()