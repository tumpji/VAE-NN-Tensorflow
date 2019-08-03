# VAE-NN-Tensorflow
Online Learning with the Variational Autoencoder and the Neural Network.

## Models
This folder contains varios models that can be used in training.
All models are implemented using Tensorflow, so gpu can be used in order to speedup training.

* K-Nearest Neighbors in KNN.py (Tensorflow v 2.0b)
* Neural Netwok in NN.py (Tensorflow v 1.9, will be updated) 
* Neural Network in NN2.py (Current progress of implementation in TF 2.0b)
* Variational Autoencoder in VAE.py (Tensorflow 1.9, will be updated)

All models have the folowing interface:

* fit(X,[Y]) train model
* predict(X) returns classification based on actual training result

## Data manipulation
Class DatasetGeneration provides interface to training.
It implements two types of iterators:

* Chunk iterator which returns data associated for each week.
* Queue iterator which returns data contents of each weeks queue with some size

## Training
Training can be divided based on actual data manipulation.

### Offline Learning
Model is trained only in first week, then is evaluated.
This is usualy unefficient in online data, because of data drift.

### Online Learning with The Last Samples
Another method is to learn online with M last samples actualized in each week.

### Online Learning with The Defayed Samples 
The last method use decay of importance of samples realized in weighs of this samples. 







