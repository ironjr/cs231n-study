# simple.py
# Jaerin Lee

import torch
import numpy as np
from scipy.misc import imsave

# Relative direction of CIFAR-10 dataset
# Original data retrieved from http://www.cs.toronto.edu/~kriz/cifar.html
cifar_dir = '../ML-Dataset/CIFAR-10/cifar-10-batches-py/'

# Refer to instruction of depackaging the dataset from the reference
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Concatenate all 5 + 1 dataset into a one single batch
xs = []
ys = []
for j in range(5):
    d = unpickle(cifar_dir + 'data_batch_' + str(j + 1))
    x = d[b'data']
    y = d[b'labels']
    xs.append(x)
    ys.append(y)
d = unpickle(cifar_dir + 'test_batch')
xs.append(d[b'data'])
ys.append(d[b'labels'])
x = np.concatenate(xs)
y = np.concatenate(ys)
x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))

'''
for i in range(50):
    imsave('cifar10_batch_' + str(i) + '.png', x[1000*i:1000*(i+1), :])
imsave('cifar10_batch_' + str(50) + '.png', x[50000:51000, :]) # test set

# dump the labels
L = 'var labels=' + str(list(y[:51000])) + ';\n'
open('cifar10_labels.js', 'w').write(L)
'''

import knn

nn = knn.NearestNeighbor()
nn.train()
