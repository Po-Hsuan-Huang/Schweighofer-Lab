#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 23:33:00 2018

@author: pohsuanh
"""
# Common imports
import numpy as np
import sys, os
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial
# to make this notebook's output stable across runs

RESUME = False

n_digits = 40
n_epochs = 2
batch_size = 150
SAVED_PATH ='/home/pohsuanh/Documents/Schweighofer Lab/variational_cifar.ckpt'
    
n_inputs = 32 * 32 * 3
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 20  # codings
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs
learning_rate = 0.001


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#x_whiten = np.sum( x_train.astype(np.float16), axis = 3 )/3
x_squash = np.divide(x_train.astype(np.float16),255)
num_examples = x_train.shape[0]
def get_next( X , batch_size ):
    global x_last_batch_index
    
    assert batch_size <= X.shape[0] # assume X a 4D tensor (n_samples, width, height, channels)
    next_batch = np.roll(x_squash, batch_size, axis = 0)[:batch_size]
    return next_batch
    
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)   
    
if os.path.isfile(SAVED_PATH + '.meta') :    
    RESUME= True
else :     
    reset_graph()
    
#    n_inputs = 32 * 32 * 3
#    n_hidden1 = 500
#    n_hidden2 = 500
#    n_hidden3 = 20  # codings
#    n_hidden4 = n_hidden2
#    n_hidden5 = n_hidden1
#    n_outputs = n_inputs
#    learning_rate = 0.001
    
    initializer = tf.contrib.layers.variance_scaling_initializer()
    my_dense_layer = partial(
        tf.layers.dense,
        activation=tf.nn.elu,
        kernel_initializer=initializer)
    
    X = tf.placeholder(tf.float32, [None, n_inputs], name = 'input')
    hidden1 = my_dense_layer(X, n_hidden1, name='h1')
    hidden2 = my_dense_layer(hidden1, n_hidden2, name='h2')
    hidden3_mean = my_dense_layer(hidden2, n_hidden3, activation=None, name='h3')
    hidden3_gamma = my_dense_layer(hidden2, n_hidden3, activation=None, name='latent_sapce')
    noise = tf.random_normal(tf.shape(hidden3_gamma), dtype=tf.float32, name='noise')
    hidden3 = hidden3_mean + tf.exp(0.5 * hidden3_gamma) * noise
    hidden4 = my_dense_layer(hidden3, n_hidden4, name='h4')
    hidden5 = my_dense_layer(hidden4, n_hidden5, name='h5')
    logits = my_dense_layer(hidden5, n_outputs, activation=None , name='logits')
    outputs = tf.sigmoid(logits, name='outputs')
    
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits, name='entropy')
    reconstruction_loss = tf.reduce_sum(xentropy, name='recon_error')
    latent_loss = 0.5 * tf.reduce_sum(
        tf.exp(hidden3_gamma) + tf.square(hidden3_mean) - 1 - hidden3_gamma, name='latent_loss')
    loss = reconstruction_loss + latent_loss
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()


with tf.Session() as sess:
    
    print('initailize graph variables')
    
    if os.path.isfile(SAVED_PATH + '.meta') :
            
        print('load session checkpoint...')
        
        saver = tf.train.import_meta_graph(SAVED_PATH + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(SAVED_PATH)))
        
    else:
        print('no checkpont found, initialize variables...')
        init.run()
        saver = tf.train.Saver()

            
    for epoch in range(n_epochs):
        n_batches = num_examples // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="") # not shown in the book
            sys.stdout.flush()                                          # not shown
            X_batch = get_next( x_squash , batch_size).reshape([-1, n_inputs])
            sess.run(training_op, feed_dict={X: X_batch})
        loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, reconstruction_loss, latent_loss], feed_dict={X: X_batch}) # not shown
        print("\r{}".format(epoch), "Train total loss:", loss_val, "\tReconstruction loss:", reconstruction_loss_val, "\tLatent loss:", latent_loss_val)  # not shown
        saver.save(sess, SAVED_PATH )  # not shown
    
    codings_rnd = np.random.normal(size=[n_digits, n_hidden3])
    outputs_val = outputs.eval(feed_dict={hidden3: codings_rnd})
#%%    
def plot_image(image, shape=[32, 32, 3]):
    plt.imshow(image.reshape(shape), interpolation="nearest")
    plt.axis("off")
    
def plot_multiple_images(images, n_rows, n_cols, pad=2):

    plt.figure(figsize=(20, 2.5 * n_digits//5)) # not shown in the book

    for iteration in range(n_digits):

        plt.subplot(n_digits//5, 5, iteration + 1)

        plot_image(images[iteration])   
#%% generate random samples
def generate_random_img() :        
    plt.figure(figsize=(16,50)) # not shown in the book
    for iteration in range(n_digits):
        plt.subplot(n_digits, 10, iteration + 1)
        plot_image(outputs_val[iteration])
    
#%%
def encode_decode():
    codings = hidden3
    X_eval = get_next( x_squash, n_digits).reshape([-1, n_inputs])
    print("input images")
    plot_multiple_images(X_eval.astype(np.float32), n_digits//5 ,5)
    
    
    with tf.Session() as sess:
        init.run()
        saver = tf.train.import_meta_graph(SAVED_PATH + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(SAVED_PATH)))
        print( '''Encode''' )
        codings_val = codings.eval(feed_dict={X:X_eval})

        print( '''Decode''' )    
        outputs_val = outputs.eval(feed_dict={codings: codings_val})
    
    print("output images")
    plot_multiple_images(outputs_val, n_digits//5 ,5)

#%% 
def interpolate_digits():

    print(""" Interpolate digits  """)

    n_iterations = 5

    N_DIGITS = 6

    codings = hidden3

    codings_rnd = np.random.normal(size=[N_DIGITS, n_hidden3])
    
    with tf.Session() as sess:

        saver.restore(sess, SAVED_PATH)

        target_codings = np.roll(codings_rnd, -1, axis=0)

        plt.figure(figsize=(16,16))

        for iteration in range(n_iterations + 1):

            codings_interpolate = codings_rnd + (target_codings - codings_rnd) * iteration / n_iterations

            outputs_val = outputs.eval(feed_dict={codings: codings_interpolate})

            for digit_index in range(N_DIGITS):

                plt.subplot(n_iterations + 1, N_DIGITS, digit_index + 1 + (N_DIGITS)*iteration)

                plot_image(outputs_val[digit_index])