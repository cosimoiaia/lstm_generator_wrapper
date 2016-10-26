#!/usr/bin/env python

##########################################
#
# lstm.py: Simple parametized python script to train e use an lstm network with tflearn for text generation.
#          Based on tflearn example code https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm_generator_cityname.py
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


def main():

    path = FLAGS.dataset

    # We avoid using padding and simply calculate the max lenght of our input set.
    maxlen = find_maxlenght(path)
    
    print("MaxLen = ", maxlen)
    X, Y, char_idx = textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3)
    

    # Here we define our network structure, using common used values for node dimensions and dropout

    g = tflearn.input_data(shape=[None, maxlen, len(char_idx)])
    g = tflearn.lstm(g, 512, return_seq=True)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.lstm(g, 512)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
    g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                           learning_rate=0.001)
    
    m = tflearn.SequenceGenerator(g, dictionary=char_idx,seq_maxlen=maxlen,clip_gradients=5.0,
                                  checkpoint_path='model_'+os.path.basename(path))
    
    # Let's train it
    seed = random_sequence_from_textfile(path, maxlen)
    m.fit(X, Y, validation_set=0.1, batch_size=FLAGS.batch_size, n_epoch=FLAGS.epochs, run_id=os.path.basename(path))


    # now we generate predictions according to the set temperature

    print("-- Test with temperature of %f --", FLAGS.temperature)
    print(m.generate(30, temperature=FLAGS.temperature, seq_seed=seed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple parametized lstm network for words generation')
    parser.add_argument('--dataset', type=str, required=True, default='', help='Path to the dataset file')
    parser.add_argument('--batch_size', type=int, default='128', help='How many string train on at a time')
    parser.add_argument('--epochs', type=int, default='1', help='How many epochs to train')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generating the predictions')
    FLAGS = parser.parse_args()
    main()
