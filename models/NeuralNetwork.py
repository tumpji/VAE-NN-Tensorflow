#!/usr/bin/python3
# -*- coding: utf-8 -*-
# =============================================================================
#         FILE: NeuralNetworkV2.py
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
#      CREATED: 2019 06.23.
# =============================================================================

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from MySessionGenerator import SessionGenerator
from MyChecker import NanChecker

class NeuralNetworkClassifier(SessionGenerator, NanChecker):
    ACTIVATIONS = {
            'elu': tf.nn.elu,
            'selu': tf.nn.selu,
            'softplus': tf.nn.softplus,
            'relu': tf.nn.relu,
            'tanh': tf.nn.tanh,
            'sigmoid': tf.nn.sigmoid }

    def __init__(self, **kwargs):
        super().__init__()
        self._build(**kwargs)
        self.finalize()
    
    def _build(self, *, input_dim, output_dim, 
            layers, activation, 
            batch_size, learning_rate, learning_rate_decay, learning_rate_minimum = None, 
            l1, l2, dropout, gaussian_noise_stddev, 
            batchnormalization, load=None):
        assert isinstance(input_dim, int) and input_dim > 0
        assert isinstance(output_dim, int) and input_dim > 0
        assert isinstance(layers, list) and len(layers) > 0
        assert all(isinstance(x, int) for x in layers)
        assert isinstance(batch_size, int) and batch_size > 0
        self.batch_size = batch_size
        assert isinstance(learning_rate, float) and learning_rate > 0.
        assert isinstance(l2, float) and l2 >= 0.
        assert isinstance(l1, float) and l2 >= 0.
        assert isinstance(batchnormalization, bool) 


        with self.session.as_default(), self.session.graph.as_default():
            self._create_feed(input_dim, batch_size)

            self.features, self.labels = self.iterator.get_next()
            self.is_training = tf.placeholder(tf.bool, shape=[], name='IsTraining')

            regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=l1, scale_l2=l2)

            input = self.features
            for size_of_layer in layers:
                input = tf.layers.dense(input, size_of_layer, use_bias=not batchnormalization, kernel_regularizer=regularizer)
                if batchnormalization:
                    input = tf.layers.batch_normalization(input, training=self.is_training)
                input = self.ACTIVATIONS[activation](input)
                if gaussian_noise_stddev > 0.:
                    input = input + tf.cond(self.is_training, 
                            true_fn=lambda: tf.random_normal(tf.shape(input), stddev=gaussian_noise_stddev),
                            false_fn=lambda :tf.zeros_like(input))

                if dropout > 0.:
                    input = tf.layers.dropout(inputs=input, rate=dropout, training=self.is_training)

            # last layer:
            logits = tf.layers.dense(input, output_dim)
            self.output_probabilities = tf.nn.softmax(logits)
            self.output_class = tf.argmax(logits, axis=1)

            self.regularization_loss = tf.losses.get_regularization_loss()
            self.mse_loss = tf.losses.sparse_softmax_cross_entropy(self.labels, logits)
            self.loss = self.regularization_loss + self.mse_loss

            self.global_step = tf.train.create_global_step()
            if learning_rate_decay > 0.:
                learning_rate = tf.train.exponential_decay(learning_rate, global_step, 1, 1-learning_rate_decay)
            if learning_rate_minimum is not None:
                learning_rate = tf.maximum(learning_rate_minimum, learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.training = tf.train.AdamOptimizer(
                    learning_rate=learning_rate).minimize(self.loss, global_step=self.global_step)

            # zjisti jeslti se zmeni
            self.session.run(tf.global_variables_initializer())

            # protoze se neda verit, ze tam bude batch-normalizace mooving avarage
            self.all_variables = tf.global_variables()

            if load is not None:
                self.load_model_filepath(load)

    def reset_weights(self):
        raise
        # jednoduse udelat pres tf.global_variables_initializer

    def _create_feed(self, shape, batch_size):
        def normal(ds):
            return ds.batch(batch_size).prefetch(4)
        def advanced(ds):
            return normal(ds.shuffle(1000000))
        self.dataset_features_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, shape])
        self.dataset_labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])

        train_dataset = tf.data.Dataset.from_tensor_slices((self.dataset_features_placeholder, self.dataset_labels_placeholder))
        test_dataset = tf.data.Dataset.from_tensor_slices((self.dataset_features_placeholder, self.dataset_labels_placeholder))
        use_dataset = tf.data.Dataset.from_tensor_slices(self.dataset_features_placeholder)


        train_dataset = advanced(train_dataset)
        test_dataset = normal(test_dataset)
        use_dataset = normal(use_dataset)

        print('This is hack - remove it')
        use_dataset = use_dataset.map(lambda x: (x, tf.zeros(dtype='int32',shape=[tf.shape(x)[0]])))

        self.iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

        self.train_initializer = self.iterator.make_initializer(train_dataset)
        self.test_initializer = self.iterator.make_initializer(test_dataset)
        self.use_initializer = self.iterator.make_initializer(use_dataset)

    def _generator_run(self, what, feed_dict):
        ''' iteruje datasetem, vraci vysledky '''
        try:
            while True:
                yield self.session.run(what, feed_dict=feed_dict)
        except tf.errors.OutOfRangeError:
            pass

    def fit(self, X, Y, maximum_epochs=200, patience=20):
        with self.session.as_default():
            Xtrain, Xtest, Ytrain, Ytest = train_test_split(self.check(X), self.check(Y))

            def compute_test():
                # initialize validation dataset
                self.session.run(self.test_initializer, feed_dict={
                    self.dataset_features_placeholder: Xtest, self.dataset_labels_placeholder: Ytest, self.is_training:False})
                return self.check(np.mean(list(self._generator_run(self.loss, {self.is_training:False}))))
            def compute_train():
                self.session.run(self.train_initializer, feed_dict={
                    self.dataset_features_placeholder: Xtrain, self.dataset_labels_placeholder: Ytrain, self.is_training:False})
                for _, l in (self._generator_run([self.training, self.loss], {self.is_training:True})):
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



    def get_probability(self, X):
        with self.session.as_default():
            self.session.run(self.use_initializer, feed_dict={self.dataset_features_placeholder: self.check(X), self.is_training:False})
            vysledky = []
            for o in self._generator_run(self.output_probabilities, feed_dict={self.is_training:False}):
                vysledky.append(o)
            return np.concatenate(vysledky, axis=0)

    def get_label(self, X):
        with self.session.as_default():
            self.session.run(self.use_initializer, feed_dict={self.dataset_features_placeholder: self.check(X), self.is_training:False})
            vysledky = []
            for o in self._generator_run(self.output_class, feed_dict={self.is_training:False}):
                vysledky.append(o)
            return np.concatenate(vysledky, axis=0)

    def get_label_and_probability(self, X):
        with self.session.as_default():
            self.session.run(self.use_initializer, feed_dict={self.dataset_features_placeholder: self.check(X), self.is_training:False})
            vysledky_label, vysledky_probab = [], []
            for a, b in self._generator_run([self.output_class, self.output_probabilities], feed_dict={self.is_training:False}):
                vysledky_label.append(a)
                vysledky_probab.append(b)
            return np.concatenate(vysledky_label, axis=0), np.concatenate(vysledky_probab, axis=0)



    def model_variables(self):
        return dict(zip(map(lambda x: x.name, self.all_variables), self.session.run(self.all_variables, feed_dict={self.is_training:False})))
    def save_model(self, filepath):
        np.savez_compressed(filepath, **self.model_variables())
    def load_model_variables(self, variables):
        for variable in self.all_variables:
            value = variables[variable.name]
            variable.load(value, self.session)
    def load_model_filepath(self, filepath):
        data = np.load(filepath)
        self.load_model_variables(data)
