#!/usr/bin/python
# -.- coding: UTF-8 -.-

import os

import numpy as np
import tensorflow as tf

# Set enviroment variables
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')
data_dir = os.path.join(dir_path, 'data')

# Network Parameters
n_input = 200 # Size of a user vector
n_hidden = 50 #
n_out = 50 # number of hobbies


def main(_):
  # Import data
  # Not yet implemented

  # Create the model
  x = tf.placeholder(tf.float32, [None, n_input])
  W = tf.Variable(tf.zeros([n_input, n_out]))
  b = tf.Variable(tf.zeros([n_out]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, n_out])

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(config.training_iters):
    batch_xs, batch_ys = (None, None) # Not yet implemented
    sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: ,      # user vectors
                                      y_: }))   # hobbies

if __name__ == '__main__':
    tf.app.run(main=main)
