#!/usr/bin/env python

import sys
import os
from shutil import copyfile
from os.path import dirname
import h5py
import numpy as np
import matplotlib.pyplot as plt

def make_directory(path):
    """Short summary.

    Parameters
    ----------
    path : Full path to the directory

    """

    if not os.path.isdir(path):
        os.mkdir(path)
        print("Making directory: " + path)
    else:
        print("Directory already exists!")

def get_parent(filepath):
    return dirname(filepath)


def load_data(file_path, reverse_compliment=False):
    # load dataset

    dataset = h5py.File(file_path, 'r')
    X_train = np.array(dataset['x_train']).astype(np.float32)
    Y_train = np.array(dataset['y_train']).astype(np.float32)
    X_valid = np.array(dataset['x_valid']).astype(np.float32)
    Y_valid = np.array(dataset['y_valid']).astype(np.float32)
    X_test = np.array(dataset['x_test']).astype(np.float32)
    Y_test = np.array(dataset['y_test']).astype(np.float32)
    X_train = X_train.transpose(0,2,1)
    X_valid = X_valid.transpose(0,2,1)
    X_test = X_test.transpose(0,2,1)
    if reverse_compliment:
        X_train_rc = X_train[:,::-1,:][:,:,::-1]
        X_valid_rc = X_valid[:,::-1,:][:,:,::-1]
        X_test_rc = X_test[:,::-1,:][:,:,::-1]

        X_train = np.vstack([X_train, X_train_rc])
        X_valid = np.vstack([X_valid, X_valid_rc])
        X_test = np.vstack([X_test, X_test_rc])

        Y_train = np.vstack([Y_train, Y_train])
        Y_valid = np.vstack([Y_valid, Y_valid])
        Y_test = np.vstack([Y_test, Y_test])
    print("Training set sizes: ", X_train.shape)
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

def plot_training(history, keyword, outdir):
    fig, ax = plt.subplots()
    ax.plot(history.history[keyword])
    ax.plot(history.history['val_{}'.format(keyword)])
    ax.set_title('model {}'.format(keyword.upper()))
    ax.set_ylabel(keyword)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    fig.savefig(os.path.join(outdir, '{}.pdf'.format(keyword)))
