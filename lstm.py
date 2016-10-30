#!/usr/bin/env python

##########################################
#
# lstm.py: Simple parametized python script to train e use an lstm network with tflearn for text generation.
#          Based on tflearn example code https://github.com/tflearn/tflearn/blob/master/examples/nlp/
#
# Author: Cosimo Iaia <cosimo.iaia@gmail.com>
# Date: 26/10/2016
#
# This file is distribuited under the terms of GNU General Public
#
#########################################



from __future__ import absolute_import, division, print_function

import os
from six import moves
import ssl
import tflearn
import argparse
from tflearn.data_utils import *



FLAGS = None


def find_maxlenght(path):
    fd = open(path)
    longest = 0
    for line in fd.readlines():
        longest = max(longest, len(line))
    
    return longest


def save_model(model):
    model.save(FLAGS.model_file)

def load_model(model):
    model.load(FLAGS.model_file)


def main():

    path = FLAGS.dataset

    # We avoid using fixed padding and simply calculate the max lenght of our input set.
    maxlen = find_maxlenght(path)
    
    print("MaxLen = ", maxlen)
    X, Y, char_idx = textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3)
    

    # Here we define our network structure, using common used values for node dimensions and dropout

    # Input Layer
    g = tflearn.input_data(shape=[None, maxlen, len(char_idx)])

    # Create our hidden LSTM Layers from parameters
    for i in range(FLAGS.hidden_layer_size):
        g = tflearn.lstm(g, 512, return_seq=True)
        g = tflearn.dropout(g, 0.5)

    
    # Finally our last lstm layer and a fully_connected with softmax activation for the output
    g = tflearn.lstm(g, 512)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.fully_connected(g, len(char_idx), activation='softmax')

    # Let's not forget our regression!
    g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                           learning_rate=0.001)
   
    # wrap it up in a sequence generator 
    m = tflearn.SequenceGenerator(g, dictionary=char_idx,seq_maxlen=maxlen,clip_gradients=5.0,
                                  checkpoint_path='model_'+os.path.basename(path))
    
    if os.path.exists(FLAGS.model_file):
	# Load our pre-train model from file
        print("Loading model from file ", FLAGS.model_file)
        load_model(m)

    # Let's train it
    print("Training model...")
    m.fit(X, Y, validation_set=0.1, batch_size=FLAGS.batch_size, n_epoch=FLAGS.epochs, run_id=os.path.basename(path))

    # save our results
    print("Saving trained model to file ", FLAGS.model_file)
    save_model(m)

    # Generate a test result
    generate(m,maxlen)



# generate predictions according to the set temperature
def generate(model,maxlen):
    seed = random_sequence_from_textfile(FLAGS.dataset, maxlen)
    print("-- Test with temperature of %f --", FLAGS.temperature)
    print(model.generate(30, temperature=FLAGS.temperature, seq_seed=seed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple parametized lstm network for words generation')
    parser.add_argument('--dataset', type=str, required=True, default='', help='Path to the dataset file')
    parser.add_argument('--batch_size', type=int, default='128', help='How many string train on at a time')
    parser.add_argument('--epochs', type=int, default='1', help='How many epochs to train')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generating the predictions')
    parser.add_argument('--model_file', type=str, default='model.tfl', help='Path to save the model file, will be loaded if present or created')
    parser.add_argument('--hidden_layer_size', type=int, default=1, help='Number of hidden lstm layers')
    FLAGS = parser.parse_args()
    main()
