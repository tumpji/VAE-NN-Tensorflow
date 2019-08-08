# VAE-NN-Tensorflow
Online Learning with the Variational Autoencoder and the Neural Network.

## Models
This folder contains various models that can be used in training. 
All models are implemented using Tensorflow, so GPU can be used in order to speed up training.

* K-Nearest Neighbors in KNN.py (Tensorflow v 2.0b) with or without weighting by distance. Also, fuzzy-KNN when any training item can appear as incomplete sample (k=(0,1>) allowing using the decay of old information in this way. 
* Neural Network in NN.py (Tensorflow v 1.9, will be updated)
* Neural Network in NN2.py (Current progress of implementation in TF 2.0b)
* Variational Autoencoder in VAE.py (Tensorflow 1.9, will be updated)

All models have the following interface:

* fit(X,[Y]) train model
* predict(X) returns classification based on actual training result

## Data manipulation
Class DatasetGeneration provides an interface to training. It implements two types of iterators:

* Chunk iterator which returns data associated for each week.
* Queue iterator which returns data contents of each week's queue with some size

## Training
Training can be divided based on actual data manipulation.

### Offline Learning
Model is trained only in the first week, then is evaluated. 
It is usually inefficient online problems because of data drift.

### Online Learning with The Last Samples
Another method is to learn online with M last samples actualized in each week.

### Online Learning with The Defayed Samples 
The latter method use decay of importance of samples realized in weighs of this samples.







