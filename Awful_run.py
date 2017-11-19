import numpy as np
import tensorflow as tf
import os
import json
import random

with open('hobbies.json') as hobbyFile:
    hobbies = sorted([x for x in json.load(hobbyFile)])

# Set enviroment variables
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')
data_dir = os.path.join(dir_path, 'data')

# Network Parameters
n_input = 400 # Size of a user vector
n_hidden_1 = 300
n_hidden_2 = 300
n_hidden_3 = 300
n_hidden_4 = 300
n_hidden_5 = 300
n_hidden_6 = 300
n_hidden_7 = 300
n_hidden_8 = 300
n_hidden_9 = 300
n_hidden_10 = 300
n_hidden_11 = 300
n_hidden_12 = 300
n_hidden_13 = 300
n_hidden_14 = 300
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

    w3 = tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3], stddev = 0.03), name = 'w3')
    b3 = tf.Variable(tf.random_normal([n_hidden_3]), name = 'b3')

    w4 = tf.Variable(tf.random_normal([n_hidden_3,n_hidden_4], stddev = 0.03), name = 'w4')
    b4 = tf.Variable(tf.random_normal([n_hidden_4]), name = 'b4')

    w5 = tf.Variable(tf.random_normal([n_hidden_4,n_hidden_5], stddev = 0.03), name = 'w5')
    b5 = tf.Variable(tf.random_normal([n_hidden_5]), name = 'b5')

    w6 = tf.Variable(tf.random_normal([n_hidden_5,n_hidden_6], stddev = 0.03), name = 'w6')
    b6 = tf.Variable(tf.random_normal([n_hidden_6]), name = 'b6')

    w7 = tf.Variable(tf.random_normal([n_hidden_6,n_hidden_7], stddev = 0.03), name = 'w7')
    b7 = tf.Variable(tf.random_normal([n_hidden_7]), name = 'b7')

    w8 = tf.Variable(tf.random_normal([n_hidden_7,n_hidden_8], stddev = 0.03), name = 'w8')
    b8 = tf.Variable(tf.random_normal([n_hidden_8]), name = 'b8')

    w9 = tf.Variable(tf.random_normal([n_hidden_8,n_hidden_9], stddev = 0.03), name = 'w9')
    b9 = tf.Variable(tf.random_normal([n_hidden_9]), name = 'b9')

    w10 = tf.Variable(tf.random_normal([n_hidden_9,n_hidden_10], stddev = 0.03), name = 'w10')
    b10 = tf.Variable(tf.random_normal([n_hidden_10]), name = 'b10')

    w11 = tf.Variable(tf.random_normal([n_hidden_10,n_hidden_11], stddev = 0.03), name = 'w11')
    b11 = tf.Variable(tf.random_normal([n_hidden_11]), name = 'b11')

    w12 = tf.Variable(tf.random_normal([n_hidden_11,n_hidden_12], stddev = 0.03), name = 'w12')
    b12 = tf.Variable(tf.random_normal([n_hidden_12]), name = 'b12')

    w13 = tf.Variable(tf.random_normal([n_hidden_12,n_hidden_13], stddev = 0.03), name = 'w13')
    b13 = tf.Variable(tf.random_normal([n_hidden_13]), name = 'b13')
    
    w14 = tf.Variable(tf.random_normal([n_hidden_13,n_hidden_14], stddev = 0.03), name = 'w14')
    b14 = tf.Variable(tf.random_normal([n_hidden_14]), name = 'b14')

    w15 = tf.Variable(tf.random_normal([n_hidden_14,n_out], stddev = 0.03), name = 'w15')
    b15 = tf.Variable(tf.random_normal([n_out]), name = 'b15')

    hidden_1_out = tf.add(tf.matmul(x,w1),b1)
    hidden_1_out = tf.nn.sigmoid(hidden_1_out)

    hidden_2_out = tf.add(tf.matmul(hidden_1_out,w2),b2)
    hidden_2_out = tf.nn.sigmoid(hidden_2_out)

    hidden_3_out = tf.add(tf.matmul(hidden_2_out,w3),b3)
    hidden_3_out = tf.nn.sigmoid(hidden_3_out)

    hidden_4_out = tf.add(tf.matmul(hidden_3_out,w4),b4)
    hidden_4_out = tf.nn.sigmoid(hidden_4_out)

    hidden_5_out = tf.add(tf.matmul(hidden_4_out,w5),b5)
    hidden_5_out = tf.nn.sigmoid(hidden_5_out)

    hidden_6_out = tf.add(tf.matmul(hidden_5_out,w6),b6)
    hidden_6_out = tf.nn.sigmoid(hidden_6_out)

    hidden_7_out = tf.add(tf.matmul(hidden_6_out,w7),b7)
    hidden_7_out = tf.nn.sigmoid(hidden_7_out)

    hidden_8_out = tf.add(tf.matmul(hidden_7_out,w8),b8)
    hidden_8_out = tf.nn.sigmoid(hidden_8_out)

    hidden_9_out = tf.add(tf.matmul(hidden_8_out,w9),b9)
    hidden_9_out = tf.nn.sigmoid(hidden_9_out)

    hidden_10_out = tf.add(tf.matmul(hidden_9_out,w10),b10)
    hidden_10_out = tf.nn.sigmoid(hidden_10_out)

    hidden_11_out = tf.add(tf.matmul(hidden_10_out,w11),b11)
    hidden_11_out = tf.nn.sigmoid(hidden_11_out)

    hidden_12_out = tf.add(tf.matmul(hidden_11_out,w12),b12)
    hidden_12_out = tf.nn.sigmoid(hidden_12_out)

    hidden_13_out = tf.add(tf.matmul(hidden_12_out,w13),b13)
    hidden_13_out = tf.nn.sigmoid(hidden_13_out)

    hidden_14_out = tf.add(tf.matmul(hidden_13_out,w14),b14)
    hidden_14_out = tf.nn.sigmoid(hidden_14_out)

    #Define loss and optimizer

    y_ = tf.nn.sigmoid(tf.add(tf.matmul(hidden_14_out, w15), b15))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(save_relative_paths=True)
    saver.restore(sess, 'nn_model_desperate.ckpt')

    def hobbify(vector):
        result = list(sess.run(y_, feed_dict={x: np.asarray(vector)}))[0]
        hobbyVectorResult = sorted([(hobbies[i],result[i]) for i in range(len(result))], key=lambda x : x[1], reverse=True)
        print(hobbyVectorResult)
        
    hobbify([[random.uniform(-1,1) for x in range(400)]])

if __name__ == '__main__':
        tf.app.run(main=main)
