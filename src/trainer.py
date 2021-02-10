#!/usr/bin/env python
from optparse import OptionParser
import os
import shutil
import sys
import utils
import model_zoo
import numpy as np
import tensorflow as tf
from tensorflow import keras

def main():
    usage = 'usage: %prog [options] <data_dir> <model_name> <output_dir> ...'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='batch_size',
        default=64,
        help='Batch size for the model training [Default: %default]')
    parser.add_option('-p', dest='patience',
        default=20,
        help='Training patience [Default: %default]')
    parser.add_option('-l', dest='learning_rate',
        default=0.1,
        help='Learning rate [Default: %default]')
    parser.add_option('-e', dest='n_epochs',
        default=100,
        help='Training number of epochs [Default: %default]')
    parser.add_option('-o', dest='model_filename',
        default='model.h5',
        help='Filename of the model [Default: %default]')

    (options, args) = parser.parse_args()
    ########TODO:ADD THE REST OF THE parameters
    if len(args) < 3:
        parser.error('Must provide data_dir, model and output directory.')
    else:
        data_path = args[0]
        model_name = args[1]
        output_dir = args[2]

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not options.model_filename.endswith('.h5'):
        options.model_filename = options.model_filename+'.h5'

    model_path = os.path.join(output_dir, options.model_filename)
#     ####LOAD DATA
    dataset = utils.load_data(data_path)
    x_train, y_train, x_valid, y_valid, x_test, y_test = dataset
    #####TODO:REMOVE THESE LINES ONCE COMPLETED
    # x_train = x_train[:100,:,:]
    # y_train = y_train[:100,:]
    # x_valid = x_valid[:100,:,:]
    # y_valid = y_valid[:100,:]
    ######
    N, L, A = x_train.shape
    input_size = (L,A)
    n_labels = y_train.shape[1]


    if model_name=='deepsea':
        model = model_zoo.deepsea(input_size, n_labels)
    elif model_name=='basset':
        model = model_zoo.basset(input_size, n_labels)
    elif model_name=='basset_mod_dr_bn':
        model = model_zoo.basset_mod_dr_bn(input_size, n_labels)
    else:
        'Model not found'

    print(model.summary())
    # set up optimizer and metrics
    auroc = keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = keras.metrics.AUC(curve='PR', name='aupr')
    optimizer = keras.optimizers.Adam(learning_rate=options.learning_rate)
    loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy', auroc, aupr])

    # train model
    es_callback = keras.callbacks.EarlyStopping(monitor='val_auroc', #'val_aupr',#
                                                patience=options.patience,
                                                verbose=1,
                                                mode='max',
                                                restore_best_weights=False)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_auroc',
                                                  factor=0.2,
                                                  patience=options.patience,
                                                  min_lr=1e-7,
                                                  mode='max',
                                                  verbose=1)

    history = model.fit(x_train, y_train,
                        epochs=int(options.n_epochs),
                        batch_size=int(options.batch_size),
                        shuffle=True,
                        validation_data=(x_valid, y_valid),
                        callbacks=[es_callback, reduce_lr])

    model.save(model_path)
    utils.plot_training(history, 'aupr', output_dir)
    utils.plot_training(history, 'auroc', output_dir)

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
