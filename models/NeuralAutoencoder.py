#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# =============================================================================
#         FILE: NeuralNetwork.py
#  DESCRIPTION:
#        USAGE:
#      OPTIONS:
# REQUIREMENTS:
#
#      LICENCE:
#
#         BUGS:
#        NOTES:
#       AUTHOR: Jiří Tumpach (tumpji),
# ORGANIZATION:
#      VERSION: 1.0
#      CREATED: 2019 04.07.
# =============================================================================

import os
import h5py 
import numpy as np
import math
import operator, functools, itertools

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from sklearn.model_selection import train_test_split
from MySessionGenerator import SessionGenerator
from MyChecker import NanChecker

class VAE(SessionGenerator, NanChecker):
    def __init__(self, **kwargs):
        super().__init__()
        self._build(**kwargs)
        self.finalize()

    def _build(self, loss, input_dim, layers, batch_size, learning_rate=0.001, 
            l1=0., l2=0., batchnormalization=False, load=None ):
        assert isinstance(input_dim, int) and input_dim > 0
        assert isinstance(layers, list) and len(layers) > 0
        assert all(isinstance(x, int) for x in layers)
        assert layers[0] > 0
        assert isinstance(batch_size, int) and batch_size > 0
        self.batch_size = batch_size

        assert isinstance(learning_rate, float) and learning_rate > 0.

        assert isinstance(l2, float) and l2 >= 0.
        assert isinstance(l1, float) and l2 >= 0.

        assert isinstance(batchnormalization, bool) 

        with self.session.as_default(), self.session.graph.as_default():
            self.create_feed(input_dim, batch_size)

            #self.input = tf.placeholder(tf.float32, shape=[None, input_dim], name='Input')
            self.input = self.iterator.get_next()
            self.is_training = tf.placeholder(tf.bool, shape=[], name='IsTraining')

            false = tf.constant(False, shape=[], dtype=tf.bool)
            self.is_sampling = tf.placeholder_with_default(false, shape=[], name='IsSampling')
            self.sample_size = tf.placeholder(tf.int32, shape=[], name='SampleSize')

            regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=l1, scale_l2=l2)

            input = self.input
            for layer in layers[:-1]:
                input = tf.layers.dense(input, layer, kernel_regularizer=regularizer, use_bias=not batchnormalization)
                if batchnormalization:
                    input = tf.layers.batch_normalization(input, training=self.is_training)
                input = tf.nn.elu(input)

            # zde je vypocet codingu
            noise = tf.random_normal(shape=[tf.shape(self.input)[0], layers[-1]])
            self.coddings_mean = tf.layers.dense(input, layers[-1], activation=None, kernel_regularizer=regularizer)
            self.coddings_logvar = tf.layers.dense(input, layers[-1], activation=None, kernel_regularizer=regularizer)

            self.codings = self.coddings_mean + tf.exp(0.5*self.coddings_logvar)*noise

            input1 = self.codings
            input2 = tf.random_normal(shape=[self.sample_size, layers[-1]])

            for layer in reversed(layers[:-1]):
                al = tf.layers.Dense(layer, kernel_regularizer=regularizer, use_bias=not batchnormalization)
                if batchnormalization:
                    bl = tf.layers.BatchNormalization()
                else:
                    bl = lambda x, **kw: x

                input1 = tf.nn.elu(bl(al(input1), training=self.is_training))
                input2 = tf.nn.elu(bl(al(input2), training=False))

            final_output = tf.layers.Dense(input_dim, kernel_regularizer=regularizer)
            self.output = final_output(input1)
            self.output_sampling = final_output(input2)

            self.regularization_loss = tf.losses.get_regularization_loss()
            self.kl_loss = -0.5*tf.reduce_sum(1 + self.coddings_logvar - tf.square(self.coddings_mean) - tf.exp(self.coddings_logvar), axis=1) 

            if loss == 'mse':
                self.reconstruction_loss = tf.losses.mean_squared_error(self.input, self.output, 
                        reduction=tf.losses.Reduction.NONE)
            elif loss == 'mae':
                self.reconstruction_loss = tf.losses.absolute_difference(self.input, self.output, 
                        reduction=tf.losses.Reduction.NONE)
            else:
                raise

            self.reconstruction_loss= tf.reduce_sum(self.reconstruction_loss, axis=1)

            self.vae_loss_sum = self.regularization_loss + tf.reduce_sum(self.reconstruction_loss + self.kl_loss)
            vae_loss = tf.truediv(self.vae_loss_sum, tf.cast(tf.shape(self.output)[0], dtype=tf.float32))

            self.global_step = tf.train.create_global_step()

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.training = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(vae_loss, global_step=self.global_step)

            # zjisti jeslti se zmeni
            self.session.run(tf.global_variables_initializer())

            # protoze se neda verit, ze tam bude batch-normalizace mooving avarage
            self.all_variables = tf.global_variables()

            if load is not None:
                self.load_model_filepath(load)

    def generate(self, N):
        ''' generate new values from random noise '''
        output = []
        while N: 
            g = self.session.run(self.output_sampling, {
                self.is_training:False, self.sample_size:min(N, self.batch_size)
                })
            output.append(self.check(g))
            N -= min(N, self.batch_size)
        return np.concatenate(output, axis=0)

    def create_feed(self, shape, batch_size):
        def normal(ds):
            return ds.batch(batch_size).prefetch(4)
        def advanced(ds):
            return normal(ds.shuffle(1000000))
        self.dataset_train_palaceholder = tf.placeholder(dtype=tf.float32, shape=[None, shape])
        self.dataset_test_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, shape])

        train_dataset = tf.data.Dataset.from_tensor_slices(self.dataset_train_palaceholder)
        test_dataset = tf.data.Dataset.from_tensor_slices(self.dataset_test_placeholder)
        train_dataset = advanced(train_dataset)
        test_dataset = normal(test_dataset)

        self.iterator = tf.data.Iterator.from_structure(
                train_dataset.output_types, train_dataset.output_shapes)

        self.train_initializer = self.iterator.make_initializer(train_dataset)
        self.test_initializer = self.iterator.make_initializer(test_dataset)

    def _generator_run(self, what, feed_dict):
        try:
            while True:
                yield self.session.run(what, feed_dict=feed_dict)
        except tf.errors.OutOfRangeError:
            pass


    def fit(self, X, maximum_epochs=200, patience=20):
        with self.session.as_default():
            Xtrain, Xtest = train_test_split(self.check(X))

            def compute_test():
                # initialize validation dataset
                self.session.run(self.test_initializer, feed_dict={self.dataset_test_placeholder: Xtest})
                return self.check(np.mean(list(self._generator_run(self.vae_loss_sum, {self.is_training: False, self.is_sampling: False}))))
            def compute_train():
                self.session.run(self.train_initializer, feed_dict={self.dataset_train_palaceholder: Xtrain})
                for _, l in (self._generator_run([self.training, self.vae_loss_sum], {self.is_training: True, self.is_sampling: False})):
                    self.check(l)
                
            minimum_loss = compute_test()
            minimum_loss_patience = patience
            best_values = self.model_variables()

            for i in range(maximum_epochs):
                compute_train()
                loss = compute_test()

                if loss < minimum_loss:
                    minimum_loss = loss
                    minimum_loss_patience = patience
                    best_values = self.model_variables()
                    print("\tBetter: ({})  {}".format(minimum_loss_patience, loss))
                elif minimum_loss_patience == 0:
                    print("\tWorse - ended {}".format(loss))
                    self.load_model_variables(best_values)
                    break
                else:
                    print("\tWorse:  ({})  {}".format(minimum_loss_patience, loss))
                    minimum_loss_patience -= 1

    def predict_codings(self, X):
        with self.session.as_default():
            self.session.run(self.test_initializer, feed_dict={self.dataset_test_placeholder: self.check(X)})
            cod, var, men = [], [], []
            for c, v, m in self._generator_run([self.codings, self.coddings_logvar, self.coddings_mean], 
                    {self.is_training: False, self.is_sampling: False}):
                cod.append(c); var.append(v); men.append(m)
            return np.concatenate(men, axis=0), np.concatenate(var, axis=0), np.concatenate(cod, axis=0)


    def predict_outputs_from_codings(self, X):
        return self.session.run([self.output], feed_dict={self.codings: self.check(X)})



    def model_variables(self):
        return dict(zip(map(lambda x: x.name, self.all_variables), self.session.run(self.all_variables)))
    def save_model(self, filepath):
        np.savez_compressed(filepath, **self.model_variables())
    def load_model_variables(self, variables):
        for variable in self.all_variables:
            value = variables[variable.name]
            variable.load(value, self.session)
    def load_model_filepath(self, filepath):
        data = np.load(filepath)
        self.load_model_variables(data)
        
