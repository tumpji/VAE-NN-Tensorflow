#!/usr/bin/python3
# -*- coding: utf-8 -*-
# =============================================================================
#         FILE: KNN.py
#  DESCRIPTION: TF2 implementation of KNN
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
#      CREATED: 2019 07.28.
# =============================================================================

from abc import abstractmethod
from enum import Enum

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input

import unittest

class Logger:
    def __init__(self, verbose=0):
        self.verbose_level = verbose
    def log(self, level, msg):
        if self.verbose_level > level:
            print(msg)

class Classifier(Logger, tf.keras.Model):
    def __init__(self, **kwargs):
        Logger.__init__(self, **kwargs)
        tf.keras.Model.__init__(self)

    @abstractmethod
    def fit(X, Y):
        pass

    @abstractmethod
    def predict(X):
        pass


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

    class WEIGHTING (Enum):
        # all k elements are weighted exactly the same, if is not possible to decide the lowest index wins
        ONE = 1       
        # all k elements are weighted based on 1/DISTANCE
        # if DISTANCE == 0, then for this particular case it switch to <ONE>
        INVERSE = 2   
        # same as <INVERSE> but when multiple DISTANCE == 0, then choose the one with lowest index
        INVERSE_WITHOUT_CORRECTION = 3 

        @property
        def exact_distance(self):
            # if this weighting needs exactly distance or it is ordering enough
            return self in [KNN.WEIGHTING.INVERSE, KNN.WEIGHTING.INVERSE_WITHOUT_CORRECTION]


    PRECISION=tf.float32
    LABEL_TYPE=tf.int32

    def __init__(self, input_dimension, number_of_labels, k, metric=None, weighted=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.input_dimension = input_dimension
        self.number_of_labels = number_of_labels
        self.k = k
        self.weighted = weighted if weighted is not None else KNN.WEIGHTING.ONE
        self.metric = metric if metric is not None else KNN.METRIC.EUCLID
        self.debug = debug

        self.train_features = tf.Variable(trainable=False, name='trainSamples',
                initial_value=np.array([[]]),
                dtype=self.PRECISION, shape=tf.TensorShape((1,self.input_dimension,None)))
        self.train_labels  = tf.Variable(trainable=False, name='trainLabels',
                initial_value=np.array([]),
                dtype=self.LABEL_TYPE, shape=tf.TensorShape((None,)),
                )

    @tf.function
    def _distance(self, X):
        # [N,F,M]
        raw_diff = tf.subtract(tf.expand_dims(X, axis=2), self.train_features)

        if self.metric == KNN.METRIC.MANHATTAN:
            distance = tf.reduce_sum(tf.abs(raw_diff), axis=1)
        elif self.metric == KNN.METRIC.EUCLID:
            distance = tf.reduce_sum(tf.square(raw_diff), axis=1)
            if self.weighted.exact_distance:
                distance = tf.sqrt(distance)
        elif self.metric._value_ < 1000:
            distance = tf.reduce_sum(tf.pow(tf.abs(raw_diff), self.metric.minkowski_power), axis=1)
            if self.weighted.exact_distance:
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

            #import pdb; pdb.set_trace();

            distance = tf.matmul(normalized_input_features, tf.squeeze(normalized_trainded_features, axis=0)) 
        else:
            raise ValueError('Unknow option KNN.METRIC')
        return distance


    @tf.function
    def _get_voting_and_values(self, X):
        distance = self._distance(X)

        distance, index_of_best = tf.nn.top_k(-distance, k=self.k, sorted=False)
        voting = tf.gather(self.train_labels, index_of_best)
        # repair for distance - its negated, but due to floating errors can be possitive too
        distance = tf.abs(distance)
        return voting, distance

    @tf.function
    def predict(self, X):
        vote, distance = self._get_voting_and_values(tf.dtypes.cast(X, self.PRECISION))
        # (N, k, classes)
        ind_vote = tf.gather(tf.eye(self.number_of_labels, dtype=self.PRECISION), vote, axis=0)

        if self.weighted == KNN.WEIGHTING.ONE:
            val_vote = ind_vote
        elif self.weighted == KNN.WEIGHTING.INVERSE:
            distance_weights = 1 / distance
            distance_weights_inf_each_k = tf.math.is_inf(distance_weights)
            distance_weights_inf = tf.reduce_any(distance_weights_inf_each_k, axis=1, keepdims=True)

            distance_weights_inf_each_k = tf.dtypes.cast(distance_weights_inf_each_k, dtype=self.PRECISION)
            distance_weights = tf.where(distance_weights_inf, distance_weights_inf_each_k, distance_weights)

            val_vote = tf.multiply(ind_vote, tf.expand_dims(distance_weights, 2))
        elif self.weighted == KNN.WEIGHTING.INVERSE_WITHOUT_CORRECTION:
            distance_weights = 1. / distance
            val_vote = tf.multiply(ind_vote, tf.expand_dims(distance_weights, 2))
        else:
            raise NotImplementedError('Implenetation of this weighting is missing')

        val_vote = tf.reduce_sum(val_vote, axis=1)

        return tf.argmax(val_vote, axis=1)

    def fit (self, X, Y):
        self.log(1, "Fitting KNN")
        self.train_labels.assign(
                tf.dtypes.cast(Y, dtype=self.LABEL_TYPE)
                )
        self.train_features.assign(
                tf.expand_dims(tf.transpose(tf.dtypes.cast(X, dtype=self.PRECISION)), axis=0)
                )



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

            k = KNN(3,6,k=3,metric=KNN.METRIC.MANHATTAN,weighted=KNN.WEIGHTING.INVERSE)
            k.fit(X,Y)

            Xtest = np.array([[0, 1, 0], [1,2,1], [3,3,1], [4,4,4], [5,10,5]], dtype=np.float32)
            Ytarget = np.array([0, 1, 3, 4, 5])
            Yres = k.predict(Xtest)

            self.assertTrue(np.all(np.equal(Yres.numpy(), Ytarget)))
            
            #### EUL
            k = KNN(3,6,k=3,metric=KNN.METRIC.EUCLID,weighted=KNN.WEIGHTING.INVERSE)
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
        Y = np.random.randint(0,3, 1000)
        Xtest = np.random.randn(*X.shape)*0.5 + X

        def target_function(self, k, weights_sklearn, metric_sklearn, metric_p_sklearn, metric_knn, weights_knn):
            target_knn = KNeighborsClassifier(
                    n_neighbors=k, weights=weights_sklearn, 
                    algorithm='auto', 
                    metric=metric_sklearn, p=metric_p_sklearn)
            target_knn.fit(X,Y)
            target_prediction = target_knn.predict(Xtest)

            test_knn = KNN(4,3,
                    k=k,metric=metric_knn, 
                    weighted=weights_knn, debug=True)
            test_knn.fit(X,Y)
            test_prediction = test_knn.predict(Xtest)

            self.assertTrue( np.all(np.equal(target_prediction, test_prediction)))

        for metric_name, metric_knn, metric_sklearn, metric_p_sklearn in [
                ('manh', KNN.METRIC.MANHATTAN,      'minkowski', 1),
                ('eucl', KNN.METRIC.EUCLID,         'minkowski', 2),
                ('maxi', KNN.METRIC.MAXIMUM,        'chebyshev', 0),
                #('cosi', KNN.METRIC.COSINE,         sklearn.metrics.pairwise.cosine_similarity, 0),
                ('omin', KNN.METRIC.MINKOWSKI_3,    'minkowski', 3) ]:
            for weights_knn, weights_sklearn in [
                    (KNN.WEIGHTING.ONE, 'uniform'), 
                    (KNN.WEIGHTING.INVERSE_WITHOUT_CORRECTION, 'distance')]:
                for k in [1,4,5]:

                    target_name = 'test_0_{}_{}_{}'.format(metric_name, weights_sklearn, k) 

                    setattr(cls, target_name, 
                            functools.partialmethod(target_function, k, weights_sklearn, metric_sklearn, metric_p_sklearn, metric_knn, weights_knn))

    add_all_test_compare_with_sklearn(TestKNN)
    unittest.main(verbosity=1)



