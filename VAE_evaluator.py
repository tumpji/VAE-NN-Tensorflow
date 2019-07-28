#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# =============================================================================
#         FILE: kNN.py
#  DESCRIPTION: Optimize kNN model using GPyOpt
#        USAGE: run as a python3.6 script 
# REQUIREMENTS: numpy, sklearn, boto3
#
#      LICENCE:
#
#         BUGS:
#        NOTES:
#       AUTHOR: Jiří Tumpach (tumpji),
# ORGANIZATION:
#      VERSION: 1.0
#      CREATED: 2019 03.25.
# =============================================================================
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import normalize, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, PowerTransformer
from MyDataScaling import MinMaxScaler

import os
import re

from dataloader import DatasetGenerator

import time, datetime, argparse

from NeuralAutoencoderV2 import VAE
from NeuralNetworkV2 import NeuralNetworkClassifier
from sklearn.externals import joblib 

import MyChecker

'''
def create_new_NN(load=None):
    raise
    return NeuralNetworkClassifier(
            input_dim=540, output_dim=2, layers=[69,32,30,11], 
            activation='tanh', batch_size=4000,
            learning_rate=0.0016145296, learning_rate_decay=0.04490795,
            l1=0.0001614529610, l2=0.092285573,
            dropout=0., gaussian_noise_stddev=0., batchnormalization=True,
            use_amsgrad=False, alpha=0.0006634615, verbose=0)

def create_new_VAE(load=None):
    raise
    return VAE(loss='mae', 
            input_dim=540, 
            layers=[200,200,10], # moje
            activation='elu',
            batch_size = 4000,
            learning_rate = 0.001097349141477545,
            learning_rate_decay = 0.008936921887503002,
            l1 = 0.,
            l2 = 0.005868939013999231, 
            dropout = 0.,
            gaussian_noise_stddev = 0.,
            batchnormalization = True,
            use_amsgrad = True, load=load
            )
'''


def create_new_NN_V2(load=None):
    return NeuralNetworkClassifier(
            input_dim=540, output_dim=2, layers=[354,322,316,305], 
            activation='relu', batch_size=730,
            learning_rate=0.004, 
            l1=0.01, l2=0.9, dropout=0.227, gaussian_noise_stddev=0.795,
            batchnormalization=True, load=load)

def create_new_VAE_V2(load=None):
    return VAE(loss='mae', 
            input_dim=540, 
            layers=[500,500,500,10], # moje
            batch_size = 1000,
            learning_rate = 0.001097349141477545,
            l1 = 0.,
            l2 = 0.005868939013999231, 
            batchnormalization = True,
            load=load
            )
################################################################################
######################## Options ###############################################
################################################################################
#/storage/plzen1/home/tumpji

class MySaver:
    def __init__(self, run_id):
        #self.server = '/storage/plzen1/home/tumpji'
        self.server = './'
        self.main_folder = self.server + 'results/VAE_cor2'

        self.temporary_results = self.main_folder + '/tmp'
        self.temporary_results_probs = self.main_folder + '/tmp_prob'
        self.temporary_network_saves = self.main_folder + '/saves'

        os.makedirs(self.temporary_results, exist_ok=True)
        os.makedirs(self.temporary_results_probs, exist_ok=True)
        os.makedirs(self.temporary_network_saves, exist_ok=True)
        self.run_id = run_id

    def save_result(self, week_id, accuracy, probabilities):
        """ save result """
        name = "/{}_{}.npy".format(self.run_id, week_id)
        np.save(self.temporary_results + name, accuracy)

        np.savez_compressed(self.temporary_results_probs + name, 
                accuracy=accuracy, probabilities=probabilities)
        print("\tResult saved {}/{}".format(self.run_id, week_id))

    def save_models(self, week_id, nn, vae, scaler):
        name = "/{}_{}_nn.npz".format(self.run_id, week_id)
        nn.save_model(self.temporary_network_saves + name)

        name = "/{}_{}_vae.npz".format(self.run_id, week_id)
        vae.save_model(self.temporary_network_saves + name)

        name = "/{}_{}_scaler.npz".format(self.run_id, week_id)
        joblib.dump(scaler, self.temporary_network_saves + name)

        print("\tResult saved {}/{}".format(self.run_id, week_id))

    def _check_if_its_loadable(self):
        regexp1 = re.compile(r'^{}_(\d+)_nn\.npz$'.format(self.run_id))
        regexp2 = re.compile(r'^{}_(\d+)_vae\.npz$'.format(self.run_id))
        regexp3 = re.compile(r'^{}_(\d+)_scaler\.npz$'.format(self.run_id))

        found_nn, found_vae, found_scaler = [], [], []
        for filename in os.listdir(self.temporary_network_saves):
            g1, g2, g3 = regexp1.match(filename), regexp2.match(filename), regexp3.match(filename)
            if g1: found_nn.append( int(g1.group(1)) )
            elif g2: found_vae.append( int(g2.group(1)) )
            elif g3: found_scaler.append( int(g3.group(1)) )

        found = sorted(list(set(found_nn).intersection( set(found_vae), set(found_scaler) )))
        print("\tFound {} saves:".format(len(found)))
        return reversed(found[-3:])

    def load_result(self):
        errors = 0
        for week_id in self._check_if_its_loadable():
            try:
                name = "/{}_{}_nn.npz".format(self.run_id, week_id)
                last_week_nn = create_new_NN_V2(load= self.temporary_network_saves + name)
                name = "/{}_{}_vae.npz".format(self.run_id, week_id)
                last_week_vae = create_new_VAE_V2(load= self.temporary_network_saves + name)
                name = "/{}_{}_scaler.npz".format(self.run_id, week_id)
                last_week_scaler = joblib.load(self.temporary_network_saves + name) 
            except Exception as e:
                errors += 1
                if errors == 3:
                    raise e
            else:
                return week_id, last_week_nn, last_week_vae, last_week_scaler
        return 0, None, None, None

# change to be object of class MySaver
saver = None

################################################################################
######################## Algorithm #############################################
################################################################################


# optimization function
def VAE_train_test(idexp):
    last_sample, last_week_nn, last_week_vae, scaler = saver.load_result()

    ds = DatasetGenerator(150000, train_iterations=1, train=False, skip_weeks=idexp)
    debug = True

    if debug:
        maximum_epochs = 1
        generate_number = 150
    else:
        maximum_epochs = 1000
        generate_number = 150000

    i = 1
    for (Xqueue, Yqueue), (Xweek, Yweek) in ds.queue_iterator():
        print('Started {}: {}'.format(i, datetime.datetime.now().strftime("%D %H:%M:%S")))
        if i <= last_sample:
            print("Skipping iteration {} (already done)".format(i))
            i += 1
            continue
        else:
            print("Started iteration no. {}".format(i))

        if i == 1:
            scaler = MinMaxScaler()
            #scaler = PowerTransformer(standardize=False)
            print('Transform')
            Xqueue = scaler.fit_transform(Xqueue)
            Xweek = scaler.transform(Xweek)

            c = MyChecker.CheckManager(3, MyChecker.NanChecker.NanError)
            while c.not_done():
                with c:
                    print('\tFit NN')
                    last_week_nn = create_new_NN_V2()
                    last_week_nn.fit(Xqueue, Yqueue, maximum_epochs=maximum_epochs)
                if c.not_done():
                    last_week_nn.free()
                    last_week_nn = None

            print('\tScore NN')
            labels, probabilities = last_week_nn.get_label_and_probability(Xweek)
            print('\tSave result')
            saver.save_result(i, np.mean(np.equal(labels, Yweek)), probabilities)

            c = MyChecker.CheckManager(3, MyChecker.NanChecker.NanError)
            while c.not_done():
                with c:
                    print('\tFit VAE')
                    last_week_vae = create_new_VAE_V2()
                    last_week_vae.fit(Xqueue, maximum_epochs=maximum_epochs)
                if c.not_done():
                    last_week_vae.free()
                    last_week_vae = None

            print('\tSave models')
            saver.save_models(i, last_week_nn, last_week_vae, scaler)
        else:
            print('\tScore with previous NN')
            labels, probabilities = last_week_nn.get_label_and_probability(scaler.transpose(Xweek))
            print('\tSave result')
            res = np.mean(np.equal(labels, Yweek))
            saver.save_result(i, res , probabilities)
            print('\t\tresult: {}'.format(res))

            c = MyChecker.CheckManager(3, MyChecker.NanChecker.NanError)
            while c.not_done():
                with c:
                    print('\tGenerate')
                    generated_data = last_week_vae.generate(generate_number)


            print('\tPredict')
            generated_prediction = last_week_nn.get_label(generated_data)
            print('\tInverse transform')
            generated_data = scaler.inverse_transform(generated_data)
            print('\tConcatenate old with new')
            Xqueue = np.concatenate([generated_data, Xqueue], axis=0)
            Yqueue = np.concatenate([generated_prediction, Yqueue], axis=0)

            print('\tTransform')
            pred = Xqueue
            Xqueue = scaler.fit_transform(Xqueue)
            np.savez('spatne.npz', tet=Xqueue, predtim=pred )

            c = MyChecker.CheckManager(3, MyChecker.NanChecker.NanError)
            while c.not_done():
                with c:
                    print('\tFit new NN')
                    new_week_nn = create_new_NN_V2()
                    new_week_nn.fit(Xqueue, Yqueue, maximum_epochs=maximum_epochs)
                if c.not_done():
                    new_week_nn.free()
                    new_week_nn = None

            c = MyChecker.CheckManager(3, MyChecker.NanChecker.NanError)
            while c.not_done():
                with c:
                    print('\tFit VAE')
                    new_week_vae = create_new_VAE_V2()
                    new_week_vae.fit(Xqueue, maximum_epochs=maximum_epochs)
                if c.not_done():
                    new_week_vae.free()
                    new_week_vae = None


            print('\tSave models')
            saver.save_models(i, new_week_nn, new_week_vae, scaler)

            last_week_nn.free()
            last_week_vae.free()
            last_week_nn = new_week_nn
            last_week_vae = new_week_vae

            print('\tClear old models')

        i += 1






################################################################################
######################## Main         ##########################################
################################################################################


parser = argparse.ArgumentParser(description='Run multiple neural networks est.')
parser.add_argument('--id', type=int, required=True)
args = parser.parse_args()

saver = MySaver(args.id)
VAE_train_test(args.id)



