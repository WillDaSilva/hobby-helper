import numpy as np
import tensorflow as tf
import os
import json

with open('hobbies.json') as hobbyFile:
    hobbies = sorted([x for x in json.load(hobbyFile)])

# Set enviroment variables
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')
data_dir = os.path.join(dir_path, 'data')

# Network Parameters
n_input = 200 # Size of a user vector
n_hidden_1 = 250
n_hidden_2 = 250
n_out = 35

def main(_):

    global_step = tf.Variable(0, name='global_step', trainable=False)

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

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(save_relative_paths=True)
    saver.restore(sess, 'nn_model.ckpt')

    def hobbify(vector):
        result = list(sess.run(y_, feed_dict={x: np.asarray(vector)}))[0]
        print(result)
        
    hobbify([[0.5 for x in range(200)]])

if __name__ == '__main__':
        tf.app.run(main=main)
