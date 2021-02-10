from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt




def basset(inputs,exp_num,padding='valid', activation='relu'):
    """
    Implemented by Amber
    """

    model = tf.keras.models.Sequential([
    #1st conv layer
    tf.keras.layers.Conv1D(300,19,input_shape=(inputs),
                          kernel_initializer='glorot_normal',
                          padding=padding),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation(activation),
    tf.keras.layers.MaxPool1D(pool_size=3),
    #2nd conv layer
    tf.keras.layers.Conv1D(200,11,
                          kernel_initializer='glorot_normal',
                          padding=padding),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool1D(pool_size=4),
    #3rd conv layer
    tf.keras.layers.Conv1D(200,7,
                          kernel_initializer='glorot_normal',
                          padding=padding),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool1D(pool_size=4),

    tf.keras.layers.Flatten(),

    #Fully connected NN
    tf.keras.layers.Dense(1000,activation='relu',
                          kernel_initializer='glorot_normal'),
    tf.keras.layers.Dropout(0.3),
    #2nd layer
    tf.keras.layers.Dense(1000,activation='relu',
                          kernel_initializer='glorot_normal'),
    tf.keras.layers.Dropout(0.3),
    #Sigmoid
    tf.keras.layers.Dense(exp_num,kernel_initializer='glorot_normal'),
    tf.keras.layers.Activation('sigmoid')

    ])

    return model


def deepsea(inputs, exp_num,padding='same', activation='relu'):
    """
    Implemented by Amber
    """
    model = tf.keras.models.Sequential([
    #1st conv layer
    tf.keras.layers.Conv1D(320,8,input_shape=inputs,
                          kernel_initializer='glorot_normal',
                          padding=padding),
    tf.keras.layers.Activation(activation),
    tf.keras.layers.MaxPool1D(pool_size=4,strides = 4),
    tf.keras.layers.Dropout(0.2),
    #2nd conv layer
    tf.keras.layers.Conv1D(480,8,
                          kernel_initializer='glorot_normal',
                          padding=padding),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool1D(pool_size=4,strides=4),
    tf.keras.layers.Dropout(0.2),
    #3rd conv layer
    tf.keras.layers.Conv1D(960,8,
                          kernel_initializer='glorot_normal',
                          padding=padding),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Flatten(),

    #Fully connected NN
    tf.keras.layers.Dense(925,activation='relu',
                          kernel_initializer='glorot_normal'),

    #Sigmoid
    tf.keras.layers.Dense(exp_num,
                          kernel_initializer='glorot_normal'),
    tf.keras.layers.Activation('sigmoid')

    ])

    #complie with optimizer
    auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy', auroc, aupr])

    return model



def basset_mod_dr_bn(inputs,exp_num,padding='valid', activation='relu'):
    """
    Implemented by Amber
    """

    model = tf.keras.models.Sequential([
    #1st conv layer
    tf.keras.layers.Conv1D(300,19,input_shape=(inputs),
                          kernel_initializer='glorot_normal',
                          padding=padding),
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation(activation),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPool1D(pool_size=3),
    #2nd conv layer
    tf.keras.layers.Conv1D(200,11,
                          kernel_initializer='glorot_normal',
                          padding=padding),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPool1D(pool_size=4),
    #3rd conv layer
    tf.keras.layers.Conv1D(200,7,
                          kernel_initializer='glorot_normal',
                          padding=padding),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPool1D(pool_size=4),

    tf.keras.layers.Flatten(),

    #Fully connected NN

    tf.keras.layers.Dense(1000,activation='linear',
                          kernel_initializer='glorot_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.3),
    #2nd layer
    tf.keras.layers.Dense(1000,activation='linear',
                          kernel_initializer='glorot_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.3),
    #Sigmoid
    tf.keras.layers.Dense(exp_num,kernel_initializer='glorot_normal'),
    tf.keras.layers.Activation('sigmoid')

    ])

    return model
