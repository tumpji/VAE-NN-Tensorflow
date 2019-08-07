#!/usr/bin/python3
# -*- coding: utf-8 -*-
# =============================================================================
#         FILE: KNN.py
#  DESCRIPTION: TF 2.0b implementation of KNN
#        USAGE: use interface of KNN class and its METRIC/WEIGHTING subclasses
#      OPTIONS:
# REQUIREMENTS:
#
#      LICENCE:
#
#         BUGS: 
#           1) Some parameters are treated as variables, this creates unnesesary large graph (tf.cond).
#           2) Cosine similarity can be optimized with some precomputations
#           3) Use optimization - like ball tree
#         TODO:
#           1) 
#
#        NOTES: 
#       AUTHOR: Jiří Tumpach (tumpji),
# ORGANIZATION:
#      VERSION: 1.0
#      CREATED: 2019 07.28.
# =============================================================================

from abc import abstractmethod
from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input


import unittest
from tools import *



class KNN(Classifier):
    class METRIC (Enum):
        MANHATTAN = 1
        MINKOWSKI_1 = 1

        EUCLID = 2
        MINKOWSKI_2 = 2

        MINKOWSKI_3 = 3
        MINKOWSKI_4 = 4
        # ... up to MINKOWSKI_999 

        MAXIMUM = 1000
        COSINE = 1001

        @property
        def minkowski_power(self):
            assert self._value_ < 1000
            return self._value_

    class WEIGHTING_BY_DISTANCE (Enum):
        # all k elements are weighted exactly the same, if is not possible to decide the lowest index wins
        NONE = 1
        ONE = 1       
        # all k elements are weighted based on 1/DISTANCE
        # if DISTANCE == 0, then for this particular case it switch to <ONE>
        INVERSE = 2   
        # same as <INVERSE> but when multiple DISTANCE == 0, then choose the one with lowest index
        INVERSE_WITHOUT_CORRECTION = 3 

        @property
        def exact_distance(self):
            # if this weighting needs exactly distance or it is ordering enough
            return self in [KNN.WEIGHTING_BY_DISTANCE.INVERSE, KNN.WEIGHTING_BY_DISTANCE.INVERSE_WITHOUT_CORRECTION]


    PRECISION=tf.float32
    LABEL_TYPE=tf.int32

    def __init__(self, input_dimension, number_of_labels, k, metric=None, weighted_by_distance=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.input_dimension = input_dimension
        self.number_of_labels = number_of_labels
        self.k = k
        self.weighted_by_distance = weighted_by_distance if weighted_by_distance is not None else KNN.WEIGHTING_BY_DISTANCE.ONE
        self.metric = metric if metric is not None else KNN.METRIC.EUCLID
        self.debug = debug

        self.train_features = tf.Variable(trainable=False, name='trainSamples',
                initial_value=np.array([[]]),
                dtype=self.PRECISION, shape=tf.TensorShape((1,self.input_dimension,None)))
        self.train_labels  = tf.Variable(trainable=False, name='trainLabels',
                initial_value=np.array([]),
                dtype=self.LABEL_TYPE, shape=tf.TensorShape((None,)),
                )
        self.train_weights = tf.Variable(trainable=False, name='trainWeights',
                initial_value=np.array([]),
                dtype=self.PRECISION, shape=tf.TensorShape((None,))
                )
        '''
        self.mean_of_weights = tf.Variable(trainable=False, name='trainWeightMean', initial_value=0,
                dtype=self.PRECISION, shape=tf.TensorShape(None))
        self.var_of_weights = tf.Variable(trainable=False, name='trainWeightMean', initial_value=0,
                dtype=self.PRECISION, shape=tf.TensorShape(None))
        '''

    @tf.function
    def _distance(self, X):
        # [N,F,M]
        raw_diff = tf.subtract(tf.expand_dims(X, axis=2), self.train_features)

        if self.metric == KNN.METRIC.MANHATTAN:
            distance = tf.reduce_sum(tf.abs(raw_diff), axis=1)
        elif self.metric == KNN.METRIC.EUCLID:
            distance = tf.reduce_sum(tf.square(raw_diff), axis=1)
            if self.weighted_by_distance.exact_distance:
                distance = tf.sqrt(distance)
        elif self.metric._value_ < 1000:
            distance = tf.reduce_sum(tf.pow(tf.abs(raw_diff), self.metric.minkowski_power), axis=1)
            if self.weighted_by_distance.exact_distance:
                distance = tf.pow(distance, 1/self.metric.minkowski_power) 
        elif self.metric == KNN.METRIC.MAXIMUM:
            distance = tf.reduce_max(tf.abs(raw_diff), axis=1)
        elif self.metric == KNN.METRIC.COSINE:
            size_features = tf.sqrt(tf.reduce_sum(tf.square(X), axis=1, keepdims=True))  
            normalized_input_features = tf.where(
                    tf.less(size_features, 1e-7),
                    X,
                    tf.realdiv(X, size_features))

            tf.print('This can be optimized')

            size_trained_features = tf.sqrt(tf.reduce_sum(tf.square(self.train_features), axis=1))
            normalized_trainded_features = tf.where(
                    tf.less(size_trained_features, 1e-7),
                    self.train_features,
                    tf.realdiv(self.train_features, size_trained_features))

            distance = tf.matmul(normalized_input_features, tf.squeeze(normalized_trainded_features, axis=0)) 
        else:
            raise ValueError('Unknow option KNN.METRIC')
        return distance

    def _get_voting_and_values_weighted(self, X):
        full_distance = -self._distance(X)
        #N, M = tf.shape(full_distance)[0], tf.shape(full_distance)[1]
        N, M = full_distance.get_shape()[0], full_distance.get_shape()[1]

        # full sort with indexes
        distance, index_of_best = tf.nn.top_k(full_distance, k=M, sorted=False)
        # cond
        cond = lambda lw0, lw1, lw2: tf.reduce_any(tf.less(lw0, self.k - 1e-7)) 
        # vars
        lw0_sum = tf.zeros(shape=(N,), dtype=self.PRECISION) # sum of all weights provided by vectors
        lw1_cont = tf.zeros(shape=(N,0), dtype=self.PRECISION) # actual value
        lw2_index = tf.zeros(shape=[], dtype=tf.dtypes.int32) # index in distances
        # body
        def body (lw0_sum, lw1_cont, lw2_index):
            remainding_weights = self.k - lw0_sum
            avaible_weights = tf.gather(self.train_weights, index_of_best[:,lw2_index])
            set_weights = tf.minimum(remainding_weights, avaible_weights)
            return (lw0_sum + set_weights, tf.concat([lw1_cont, tf.expand_dims(set_weights,1)], axis=1) , lw2_index + 1)

        res = tf.while_loop(
                cond = cond, body = body, loop_vars = [lw0_sum, lw1_cont, lw2_index],
                shape_invariants=[tf.TensorShape([N]), tf.TensorShape([N, None]), tf.TensorShape([])],
                maximum_iterations = M)

        size_of_vector = tf.shape(res[1])[1]

        voting = tf.gather(self.train_labels, index_of_best[:,:size_of_vector])
        voting = tf.gather(tf.eye(self.number_of_labels, dtype=self.PRECISION), voting, axis=0)
        voting = tf.multiply(voting, tf.expand_dims(res[1], axis=2))
        # (N, k, labels) [float]
        return voting, tf.abs(distance[:,:size_of_vector])


    def _get_voting_and_values_unweighted(self, X):
        full_distance = -self._distance(X)
        N = tf.shape(X)[0]
        # (N, M) [float]
        # (N, M) [int] index to neighbor
        distance, index_of_best = tf.nn.top_k(full_distance, k=self.k, sorted=False)
        # (N, M) [int] labels of neigbour
        voting = tf.gather(self.train_labels, index_of_best)
        # (N, k, labels) [float] one-hot encoding
        voting = tf.gather(tf.eye(self.number_of_labels, dtype=self.PRECISION), voting, axis=0)
        # repair for distance - its negated, but due to floating errors can be possitive too
        return voting, tf.abs(distance)



    #@tf.function
    def predict(self, X):
        X = tf.dtypes.cast(X, self.PRECISION)
        if self.use_fuzzy_weights:
            ind_vote, distance = self._get_voting_and_values_weighted(X)
        else:
            ind_vote, distance = self._get_voting_and_values_unweighted(X)

        if self.weighted_by_distance == KNN.WEIGHTING_BY_DISTANCE.ONE:
            val_vote = ind_vote
        elif self.weighted_by_distance == KNN.WEIGHTING_BY_DISTANCE.INVERSE:
            distance_weights = 1 / distance
            distance_weights_inf_each_k = tf.math.is_inf(distance_weights)
            distance_weights_inf = tf.reduce_any(distance_weights_inf_each_k, axis=1, keepdims=True)

            distance_weights_inf_each_k = tf.dtypes.cast(distance_weights_inf_each_k, dtype=self.PRECISION)
            distance_weights = tf.where(distance_weights_inf, distance_weights_inf_each_k, distance_weights)

            val_vote = tf.multiply(ind_vote, tf.expand_dims(distance_weights, 2))
        elif self.weighted_by_distance == KNN.WEIGHTING_BY_DISTANCE.INVERSE_WITHOUT_CORRECTION:
            distance_weights = 1. / distance
            val_vote = tf.multiply(ind_vote, tf.expand_dims(distance_weights, 2))
        else:
            raise NotImplementedError('Implenetation of this weighting is missing')

        val_vote = tf.reduce_sum(val_vote, axis=1)

        return tf.argmax(val_vote, axis=1)

    def fit (self, X, Y, weights=None):
        """ Train model, weights represents proportion ok full sample to which is k gatthered. """
        self.log(1, "Fitting KNN")
        self.train_labels.assign(tf.dtypes.cast(Y, dtype=self.LABEL_TYPE))
        self.train_features.assign(
                tf.expand_dims(tf.transpose(tf.dtypes.cast(X, dtype=self.PRECISION)), axis=0)
                )

        self.use_fuzzy_weights = False

        if weights is not None:
            # to ne jde ot w
            #if self.weighted is self.WEIGHTING.ONE: raise ValueError('Weighting can be used ')
            self.train_weights.assign(tf.dtypes.cast(weights, dtype=self.PRECISION))
            #self.mean_of_weights.assign(tf.reduce_mean(self.train_weights))
            self.use_fuzzy_weights = True




#************************************************************************************************************************
#*************************************************  TESTS  **************************************************************
#************************************************************************************************************************

if __name__ == '__main__':
    import sklearn
    import functools
    from sklearn import datasets
    from sklearn.neighbors import KNeighborsClassifier

    class TestKNN(unittest.TestCase):
        PRECISION = np.float16
        '''
        def __init__(self, *args, **kwargs):
            self.add_all_test_compare_with_sklearn()
            super().__init__(*args, **kwargs)
        '''

        def test_init(self):
            k = KNN(10,2,3,metric=KNN.METRIC.MANHATTAN)
        def test_fit(self):
            X = np.array([[0, 1, 2], [1, 1, 1], [2,0,2]])
            Y = np.array([0, 1, 2], dtype=np.int32)

            k = KNN(3,2,k=1,metric=KNN.METRIC.MANHATTAN)
            k.fit(X,Y)

            self.assertTrue(np.all(np.equal(k.train_features.numpy(), np.transpose(X))))
            self.assertTrue(np.all(np.equal(k.train_labels.numpy(), Y)))

        def test_distance_manhattan(self):
            def train_on(metric):
                X = np.array([[0, 1, 2], [1, 1, 1], [2,0,2]])
                Y = np.array([0, 1, 2])

                k = KNN(3,2,k=1,metric=metric)
                k.fit(X,Y)
                return k

            Xtest = np.array([[1,1,1], [2,2,0]])

            k = train_on(KNN.METRIC.MANHATTAN)
            Ytrue_manhaton = np.array([ [2, 0, 3], [5, 3, 4]] , dtype=np.float32)
            Yresult = k._distance(tf.dtypes.cast(Xtest, k.PRECISION))
            self.assertTrue(np.all(np.equal(Ytrue_manhaton, Yresult)))

            
            k = train_on(KNN.METRIC.MAXIMUM)
            Ytrue_maximum = np.array([[1, 0, 1], [2, 1, 2]])
            Yresult = k._distance(tf.dtypes.cast(Xtest, k.PRECISION))
            self.assertTrue(np.all(np.equal(Ytrue_maximum, Yresult)))

            k = train_on(KNN.METRIC.EUCLID)
            Ytrue_euclid = np.array([[2, 0, 3], [9, 3, 8]])
            Yresult = k._distance(tf.dtypes.cast(Xtest, k.PRECISION))
            self.assertTrue(np.all(np.equal(Ytrue_euclid, Yresult)))

        def test_predict_k1(self):
            X = np.array([ [3,3,3], [4,4,4], [0,0,0], [5,5,5], [1,1,1], [2,2,2] ])
            Y = np.array([3,4,0,5,1,2])

            k = KNN(3,6,k=1,metric=KNN.METRIC.MANHATTAN)
            k.fit(X,Y)

            Xtest = np.array([[0, 1, 0], [1,2,1], [3,3,1], [4,4,4], [5,10,5]])
            Ytarget = np.array([0, 1, 3, 4, 5])
            Yres = k.predict(Xtest)

            self.assertTrue(np.all(np.equal(Yres.numpy(), Ytarget)))

        def test_predict_k3(self):
            X = np.array([ [3,3,3], [4,4,4], [0,0,0], [5,5,5], [1,1,1], [2,2,2] ])
            X = np.concatenate([X, X+np.array([0,0,1]), X-np.array([0,1,0])], axis=0)
            Y = np.array([3,4,0,5,1,2])
            Y = np.concatenate([Y,Y,Y], axis=0)

            k = KNN(3,6,k=3,metric=KNN.METRIC.MANHATTAN)
            k.fit(X,Y)

            Xtest = np.array([[0, 1, 0], [1,2,1], [3,3,1], [4,4,4], [5,10,5]])
            Ytarget = np.array([0, 1, 3, 4, 5])
            Yres = k.predict(Xtest)

            self.assertTrue(np.all(np.equal(Yres.numpy(), Ytarget)))
            
            #### EUL
            k = KNN(3,6,k=3,metric=KNN.METRIC.EUCLID)
            k.fit(X,Y)

            Xtest = np.array([[0, 1, 0], [1,2,1], [3,3,1], [4,4,4], [5,10,5]])
            Ytarget = np.array([0, 1, 3, 4, 5])
            Yres = k.predict(Xtest)

            self.assertTrue(np.all(np.equal(Yres.numpy(), Ytarget)))

        def test_predict_k3_weighted(self):
            X = np.array([ [3,3,3], [4,4,4], [0,0,0], [5,5,5], [1,1,1], [2,2,2] ], dtype=np.float32)
            X = np.concatenate([X, X+np.array([0,0,1]), X-np.array([0,1,0])], axis=0)
            Y = np.array([3,4,0,5,1,2], dtype=np.int32)
            Y = np.concatenate([Y,Y,Y], axis=0)

            k = KNN(3,6,k=3,metric=KNN.METRIC.MANHATTAN,weighted_by_distance=KNN.WEIGHTING_BY_DISTANCE.INVERSE)
            k.fit(X,Y)

            Xtest = np.array([[0, 1, 0], [1,2,1], [3,3,1], [4,4,4], [5,10,5]], dtype=np.float32)
            Ytarget = np.array([0, 1, 3, 4, 5])
            Yres = k.predict(Xtest)

            self.assertTrue(np.all(np.equal(Yres.numpy(), Ytarget)))
            
            #### EUL
            k = KNN(3,6,k=3,metric=KNN.METRIC.EUCLID,weighted_by_distance=KNN.WEIGHTING_BY_DISTANCE.INVERSE)
            k.fit(X,Y)

            Xtest = np.array([[0, 1, 0], [1,2,1], [3,3,1], [4,4,4], [5,10,5]])
            Ytarget = np.array([0, 1, 3, 4, 5])
            Yres = k.predict(Xtest)

            self.assertTrue(np.all(np.equal(Yres.numpy(), Ytarget)))


        def test_cosine_distance(self):
            X = np.array([ [4,4,4], [0,0,0], [1,-1,1], [2,-2,2] ], dtype=np.float32)
            Y = np.array([3,4,0,5,1,2], dtype=np.int32)
            k = KNN(3,6,k=3,metric=KNN.METRIC.COSINE)
            k.fit(X,Y)

            target = sklearn.metrics.pairwise.cosine_similarity(X+1, X)
            test = k._distance(X+1)

            self.assertTrue(np.all(np.equal(test, target)))



    def add_all_test_compare_with_sklearn(cls):
        X = np.random.rand(1000, 4)
        Y = np.random.randint(0, 3, 1000)
        W = np.ones((2000,), dtype=np.float32) / 2
        Xtest = np.random.randn(*X.shape)*0.5 + X

        def target_function(self, k, weights_sklearn, metric_sklearn, metric_p_sklearn, metric_knn, weights_knn):
            target_knn = KNeighborsClassifier(
                    n_neighbors=k, weights=weights_sklearn, algorithm='auto', 
                    metric=metric_sklearn, p=metric_p_sklearn)
            target_knn.fit(X,Y)
            target_prediction = target_knn.predict(Xtest)

            test_knn = KNN(4,3, k=k, metric=metric_knn, 
                    weighted_by_distance=weights_knn, debug=True)
            test_knn.fit(X,Y)
            test_prediction = test_knn.predict(Xtest)

            self.assertTrue( np.all(np.equal(target_prediction, test_prediction)))

        def target_weighted_function(self, k, weights_sklearn, metric_sklearn, metric_p_sklearn, metric_knn, weights_knn):
            target_knn = KNeighborsClassifier(
                    n_neighbors=k, weights=weights_sklearn, algorithm='auto', 
                    metric=metric_sklearn, p=metric_p_sklearn)
            target_knn.fit(X,Y)
            target_prediction = target_knn.predict(Xtest)

            test_knn = KNN(4,3, k=k, metric=metric_knn, 
                    weighted_by_distance=weights_knn, debug=True)
            test_knn.fit( np.concatenate([X,X], axis=0), np.concatenate([Y,Y], axis=0), weights=W)
            test_prediction = test_knn.predict(Xtest)

            self.assertTrue( np.all(np.equal(target_prediction, test_prediction)))

        for metric_name, metric_knn, metric_sklearn, metric_p_sklearn in [
                ('manh', KNN.METRIC.MANHATTAN,      'minkowski', 1),
                ('eucl', KNN.METRIC.EUCLID,         'minkowski', 2),
                ('maxi', KNN.METRIC.MAXIMUM,        'chebyshev', 0),
                ('omin', KNN.METRIC.MINKOWSKI_3,    'minkowski', 3) ]:
            for weights_knn, weights_sklearn in [
                    (KNN.WEIGHTING_BY_DISTANCE.ONE, 'uniform'), 
                    (KNN.WEIGHTING_BY_DISTANCE.INVERSE_WITHOUT_CORRECTION, 'distance')]:
                for k in [1,4,5]:
                    target_name = 'test_0_{}_{}_{}'.format(metric_name, weights_sklearn, k) 
                    target_weighted_name = target_name + '_w'

                    setattr(cls, target_name, 
                            functools.partialmethod(target_function, k, weights_sklearn, 
                                metric_sklearn, metric_p_sklearn, metric_knn, weights_knn))

                    setattr(cls, target_weighted_name, 
                            functools.partialmethod(target_weighted_function, k, weights_sklearn, 
                                metric_sklearn, metric_p_sklearn, metric_knn, weights_knn))

    add_all_test_compare_with_sklearn(TestKNN)
    unittest.main(verbosity=1)


'''
    def _get_voting_and_values(self, X):
            @return 
                voting = [(N, self.size_features, 
        # (N, M)
        full_distance = -self._distance(X)

        # TODO: upgrade to better prediction
        maxk = tf.dtypes.cast(tf.math.ceil((self.k + 4) / self.mean_of_weights), tf.dtypes.int32)

        # (N, M) [float]
        # (N, M) [int] index to neighbor
        distance, index_of_best = tf.nn.top_k(full_distance, k=maxk, sorted=False)
        # (M, N) [float] weights of neighbors
        weights_of_best = tf.gather(self.train_weights, tf.transpose(index_of_best))

        def fce1(state, weights):
            # state = (2, N) - [sum weights, last_increse]
            sum_weights = state[0,:]
            rem = self.k - sum_weights
            pos = tf.minimum(rem, weights)
            return tf.stack([sum_weights + pos, pos])

        # (M,2,N)
        r = tf.scan(fn=fce1, elems=weights_of_best, 
                initializer=tf.zeros((2,tf.shape(weights_of_best)[1])))

        # (N, M) - weighted k
        voting = tf.transpose(r[:,1,:])


        # compare what is underrepresented
        voting_underfill = r[-1,0,:] < self.k
        underfill_needed = tf.reduce_any(voting_underfill)

        full_distance_2 = tf.boolean_mask(distance, voting_underfill, axis=0)
        distance_2, index_of_best_2 = tf.nn.top_k(full_distance_2, k=tf.shape(full_distance)[1], sorted=False)
        # (M,N)
        weights_of_best_2 = tf.gather(self.train_weights, tf.transpose(index_of_best_2))

        r_2 = tf.scan(fn=fce1, elems=weights_of_best_2, 
                initializer=tf.zeros((2,tf.shape(weights_of_best_2)[1])))
        
        maximum_index = tf.argmax(tf.dtypes.cast(tf.reduce_all(r_2[:,0,:] >= self.k, axis=0), dtype=tf.dtypes.int32))
        voting_2 = tf.concat([
                voting, 
                tf.zeros(tf.shape(voting)[0], tf.maximum(0,maximum_index + 1 -  tf.shape(voting)[1]) )
            ])
        voting_2 = tf.scatter_update(
                voting_2, 
                tf.where(voting_underfill),
                tf.transpose(r_2[:,1,:])
                )

        voting = tf.cond(underfill_needed, voting_2, voting)


        # repair for distance - its negated, but due to floating errors can be possitive too
        return voting, tf.abs(distance)
'''
