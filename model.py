# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 15:25:22 2018

@author: shuan
"""
import tensorflow as tf
import numpy as np
import keras
from pre_process import ToHdf5
import h5py

#keras训练模型，输入（模型的输入）
def heartSoundModel(X_shape):
    X_input=keras.layers.Input(X_shape)
    print(X_input)
    X=keras.layers.Conv1D(filters=32,kernel_size=12,strides=4,activation="relu",use_bias=True,name="conv_0")(X_input)
    X=keras.layers.Conv1D(filters=64,kernel_size=8,strides=2,activation="relu",use_bias=True,name="conv_1")(X)
    X=keras.layers.Conv1D(filters=64,kernel_size=6,strides=2,activation="relu",use_bias=True,name="conv_2")(X)
    X=keras.layers.Conv1D(filters=64,kernel_size=6,strides=2,activation="relu",use_bias=True,name="conv_3")(X)
    X=keras.layers.Conv1D(filters=16,kernel_size=4,strides=2,activation="relu",use_bias=True,name="conv_4")(X)
    X=keras.layers.Conv1D(filters=16,kernel_size=3,strides=2,activation="relu",use_bias=True,name="conv_5")(X)
    X=keras.layers.Flatten()(X)
    X=keras.layers.Dense(224,activation="relu",use_bias=True,name="full_0")(X)
    X=keras.layers.Dense(1,activation="sigmoid",use_bias=True,name="full_1")(X)
    model=keras.Model(inputs=X_input,outputs=X,name="heartSoundModel")
    return model


train= ToHdf5()
X_train,Y_train,b,d=train.loadData('testHdf5.hdf5')
hmodel=heartSoundModel(X_train.shape[1:])
hmodel.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
hmodel.fit(x=X_train,y=Y_train,epochs=4,batch_size=10)
#hmodel.train_on_batch(X_train,Y_train)
hmodel.save_weights("weight.hdf5")