#!/usr/bin/python
# -.- coding: UTF-8 -.-

import os
import json

import numpy as np
import tensorflow as tf

# Set enviroment variables
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')
data_dir = os.path.join(dir_path, 'data')

# Network Parameters
n_input = 200 # Size of a user vector
n_hidden_1 = 250
n_hidden_2 = 250
n_out = 0 # number of hobbies initialized after hob is loaded

batch_size = 100
learning_rate = 0.5
total_batches = 1000

def next_batch(batch_size, data, labels):
    idx = np.arange(0,len(data))
    np.random.shuffle(idx)
    idx = idx[:batchsize]
    data_shuffled = [data[i] for i in idx]
    labels_shuffled = [labels[i] for i in idx]

    return np.asarray(data_shuffled), np.asarray(labels_shuffled)

def hobbyToVector(hobbies):
    ones = []
    for hobby in hobbies:
        ones.append(hob.indexOf(hobby))
    return [0 for x in range(len(hob)) if x not in ones else 1]

def main(_):
    # Import data
    with open('vectorsAndLabels.json', 'r') as vl:
        wordVectors = np.asarray([np.asarray(x) for x in json.loads(vl.readline())])
        hobbies = np.asarray([hobbyToVector(x) for x in json.loads(vl.readline())])
    with open('hobbies.json', 'r') as hobFile:
            hob = json.load(hobFile)
            hob = list(sorted(hob.keys()))
            n_out = len(hob)

    # Create the model
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_out])

    w1 = tf.Variable(tf.random_normal([n_input,n_hidden_1], stddev = 0.03), name = 'w1')
    b1 = tf.Variable(tf.random_normal([n_hidden_1]), name = 'b1')

    w2 = tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2], stddev = 0.03), name = 'w2')
    b2 = tf.Variable(tf.random_normal([n_hidden_2]), name = 'b2')

    w3 = tf.Variable(tf.random_normal([n_hidden_2,n_out], stddev = 0.03), name = 'w3')
    b3 = tf.Variable(tf.random_normal([n_out]), name = 'b3')

    hidden_1_out = tf.add(tf.matmul(x,w1),b1)
    hidden_1_out = tf.nn.sigmoid(hidden_1_out)

    hidden_2_out = tf.add(tf.matmul(hidden_1_out,w2),b2)
    hidden_2_out = tf.nn.sigmoid(hidden_2_out)

    #Define loss and optimizer

    y_ = tf.nn.sigmoid(tf.add(tf.matmul(hidden_2_out, w3), b3))

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_))
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(total_batches):
        batch_xs, batch_ys = next_batch(batch_size, wordVectors, hobbies)
        sess.run(train, feed_dict={x: batch_xs, y: batch_ys})

    # # Test trained model
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run(accuracy, feed_dict={x: ,      # user vectors
    #                                     y_: }))   # hobbies

if __name__ == '__main__':
        tf.app.run(main=main)
