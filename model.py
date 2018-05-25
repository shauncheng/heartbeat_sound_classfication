# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 15:25:22 2018

@author: shuan
模型搭建
"""
#import tensorflow as tf
import keras
from keras import regularizers

def heartSoundModel_reg(X_shape):
    """
    l2 正则化模型
    """
    X_input=keras.layers.Input(X_shape)
    print(X_input)
    X=keras.layers.Conv1D(filters=32,kernel_size=12,strides=4,activation="relu",use_bias=True,name="conv_0",kernel_regularizer=keras.regularizers.l2(0.01))(X_input)
    X=keras.layers.Conv1D(filters=64,kernel_size=8,strides=2,activation="relu",use_bias=True,name="conv_1",kernel_regularizer=keras.regularizers.l2(0.01))(X)
    X=keras.layers.Conv1D(filters=64,kernel_size=6,strides=2,activation="relu",use_bias=True,name="conv_2",kernel_regularizer=keras.regularizers.l2(0.01))(X)
    X=keras.layers.Conv1D(filters=64,kernel_size=6,strides=2,activation="relu",use_bias=True,name="conv_3",kernel_regularizer=keras.regularizers.l2(0.01))(X)
    X=keras.layers.Conv1D(filters=16,kernel_size=4,strides=2,activation="relu",use_bias=True,name="conv_4",kernel_regularizer=keras.regularizers.l2(0.01))(X)
    X=keras.layers.Conv1D(filters=16,kernel_size=3,strides=2,activation="relu",use_bias=True,name="conv_5",kernel_regularizer=keras.regularizers.l2(0.01))(X)
    X=keras.layers.Flatten()(X)
    X=keras.layers.Dense(224,activation="relu",use_bias=True,name="full_0")(X)
    X=keras.layers.Dense(1,activation="sigmoid",use_bias=True,name="full_1")(X)
    model=keras.Model(inputs=X_input,outputs=X,name="heartSoundModel")
    return model

def heartSoundModel_drop(X_shape):
    """
    dropout 正则化模型
    """
    X_input=keras.layers.Input(X_shape)
    print(X_input)
    X=keras.layers.Conv1D(filters=32,kernel_size=12,strides=4,activation="relu",use_bias=True,name="conv_0")(X_input)
    X=keras.layers.Conv1D(filters=64,kernel_size=8,strides=2,activation="relu",use_bias=True,name="conv_1")(X)
    X=keras.layers.Dropout(0.5)(X)
    X=keras.layers.Conv1D(filters=64,kernel_size=6,strides=2,activation="relu",use_bias=True,name="conv_2")(X)
    X=keras.layers.Dropout(0.5)(X)
    X=keras.layers.Conv1D(filters=64,kernel_size=6,strides=2,activation="relu",use_bias=True,name="conv_3")(X)
    X=keras.layers.Conv1D(filters=16,kernel_size=4,strides=2,activation="relu",use_bias=True,name="conv_4")(X)
    X=keras.layers.Conv1D(filters=16,kernel_size=3,strides=2,activation="relu",use_bias=True,name="conv_5")(X)
    X=keras.layers.Flatten()(X)
    X=keras.layers.Dense(224,activation="relu",use_bias=True,name="full_0")(X)
    X=keras.layers.Dense(1,activation="sigmoid",use_bias=True,name="full_1")(X)
    model=keras.Model(inputs=X_input,outputs=X,name="heartSoundModel")
    return model