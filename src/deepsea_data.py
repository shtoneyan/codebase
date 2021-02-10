#!/usr/bin/env python
import os, sys, h5py, scipy.io
import numpy as np
import subprocess as sp
import utils
from optparse import OptionParser

def main():
    usage = 'usage: %prog [options] <data_path> <output_folder>'
    parser = OptionParser(usage)
    parser.add_option('-n', dest='N',
        default='919', type='int',
        help='Number of labels to subset [Default: %default]')
    (options,args) = parser.parse_args()
    if len(args) != 2:
        parser.error('Must provide data path and output folder')
    else:
        data_path = sys.argv[1] # dir where deepsea_train folder is
        output_folder = sys.argv[2] # output_folder to save dataset in

    utils.make_directory(output_folder)
    # base_dir = utils.get_parent(files_path)
    class_range=range(options.N)
    uncompressed_data_dir = os.path.join(data_path, 'deepsea_train')
    train = load_set('train', uncompressed_data_dir, class_range)
    valid = load_set('valid', uncompressed_data_dir, class_range)
    test = load_set('test', uncompressed_data_dir, class_range)
    save_deepsea_dataset(train, valid, test, output_folder)


def data_subset(y, class_range):
    " gets a subset of data in the class_range"
    data_index = []
    for i in class_range:
        index = np.where(y[:, i] == 1)[0]
        data_index = np.concatenate((data_index, index), axis=0)
    unique_index = np.unique(data_index)
    return unique_index.astype(int)

def load_set(data_fold, filepath, class_range):
    assert data_fold in ['train', 'test', 'valid']
    print("loading {} data".format(data_fold))
    if data_fold == 'train':
        mat = h5py.File(os.path.join(filepath, data_fold+'.mat'), 'r')
        Y = np.transpose(mat[data_fold+'data'], axes=(1,0))
        axes_order = (2,1,0)
    else:
        mat = scipy.io.loadmat(os.path.join(filepath, data_fold+'.mat'))
        Y = np.array(mat[data_fold+'data'])
        axes_order = (0,1,2)

    index = data_subset(Y, class_range)
    Y = Y[:,class_range]
    Y = Y[index,:]
    X = np.transpose(mat[data_fold+'xdata'], axes=axes_order)
    X = X[index,:,:]
    X = X[:,[0,2,1,3],:]

    # X_train = np.expand_dims(X_train, axis=3)
    return((X.astype(np.int8), Y.astype(np.int8)))


def save_deepsea_dataset(train, valid, test, output_dir):
    """ save to h5py dataset """
    print("saving dataset")
    h5f = h5py.File(os.path.join(output_dir, 'deepsea.h5'), 'w')
    h5f.create_dataset('x_train', data=train[0], dtype='int8')
    h5f.create_dataset('y_train', data=train[1], dtype='int8')
    h5f.create_dataset('x_valid', data=valid[0], dtype='int8')
    h5f.create_dataset('y_valid', data=valid[1], dtype='int8')
    h5f.create_dataset('x_test', data=test[0], dtype='int8')
    h5f.create_dataset('y_test', data=test[1], dtype='int8')
    h5f.close()

if __name__ == '__main__':
  main()
