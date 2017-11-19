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
n_hidden_1 = 300
n_hidden_2 = 300
n_hidden_3 = 300
n_out = 0 # number of hobbies initialized after hob is loaded

batch_size = 100
learning_rate = 0.5
total_batches = 1000
save_iteration = total_batches // 4

def next_batch(b_size, data, labels):
    idx = np.arange(0,len(data))
    np.random.shuffle(idx)
    idx = idx[:b_size]
    data_shuffled = [data[i] for i in idx]
    labels_shuffled = [labels[i] for i in idx]

    return np.asarray(data_shuffled), np.asarray(labels_shuffled)

with open('hobbies.json', 'r') as hobFile:
            hob = json.load(hobFile)
            hob = list(sorted(hob.keys()))
            n_out = len(hob)

def hobbyToVector(hobbies):
    ones = []
    for hobby in hobbies:
        ones.append(hob.index(hobby))
    return [0 if x not in ones else 1 for x in range(len(hob))]

def main(_):
    # Import data
    with open('vectorsAndLabels.json', 'r') as vl:
        wordVectors = np.asarray([np.asarray(x) for x in json.loads(vl.readline())])
        hobbies = np.asarray([hobbyToVector(x) for x in json.loads(vl.readline())])

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create the model
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_out])

    w1 = tf.Variable(tf.random_normal([n_input,n_hidden_1], stddev = 0.03), name = 'w1')
    b1 = tf.Variable(tf.random_normal([n_hidden_1]), name = 'b1')

    w2 = tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2], stddev = 0.03), name = 'w2')
    b2 = tf.Variable(tf.random_normal([n_hidden_2]), name = 'b2')

    w3 = tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3], stddev = 0.03), name = 'w3')
    b3 = tf.Variable(tf.random_normal([n_hidden_3]), name = 'b3')

    w4 = tf.Variable(tf.random_normal([n_hidden_3,n_out], stddev = 0.03), name = 'w4')
    b4 = tf.Variable(tf.random_normal([n_out]), name = 'b4')

    hidden_1_out = tf.add(tf.matmul(x,w1),b1)
    hidden_1_out = tf.nn.sigmoid(hidden_1_out)

    hidden_2_out = tf.add(tf.matmul(hidden_1_out,w2),b2)
    hidden_2_out = tf.nn.sigmoid(hidden_2_out)

    hidden_3_out = tf.add(tf.matmul(hidden_2_out,w3),b3)
    hidden_3_out = tf.nn.sigmoid(hidden_3_out)

    #Define loss and optimizer

    y_ = tf.nn.sigmoid(tf.add(tf.matmul(hidden_3_out, w4), b4))

    cross_entropy = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_,
            logits=y
        )
    )
    train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = train.minimize(cross_entropy, global_step=global_step)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(save_relative_paths=True)

    # Train
    for i in range(total_batches):
        if i % (total_batches // 20) == 0:
            print(i)
        batch_xs, batch_ys = next_batch(batch_size, wordVectors, hobbies)
        sess.run(train, feed_dict={x: batch_xs, y: batch_ys})
    else:
        saver.save(sess, 'nn_model.ckpt') #, global_step)

if __name__ == '__main__':
        tf.app.run(main=main)
