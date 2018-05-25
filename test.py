# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 18:12:43 2018

@author: shuan

this file is used to test other function, so ignore it!
"""
import pylab as pl  
from pre_process import AudioTool
from pre_process import ToHdf5
import h5py
import numpy as np

union=lambda x,y:np.append(x,y,axis=0)
def main():
    print('model launch')
    audioTool=AudioTool()
    audioTool.findFile('../audio_data/training/training-a','.wav')
    train_a_X=audioTool.cutAudio(4096)
    train_a_Y=audioTool.get_csv()
    audioTool.findFile('../audio_data/training/training-b','.wav')
    train_b_X=audioTool.cutAudio(4096)
    train_b_Y=audioTool.get_csv()
    audioTool.findFile('../audio_data/training/training-c','.wav')
    train_c_X=audioTool.cutAudio(4096)
    train_c_Y=audioTool.get_csv()
    audioTool.findFile('../audio_data/training/training-d','.wav')
    train_d_X=audioTool.cutAudio(4096)
    train_d_Y=audioTool.get_csv()
    audioTool.findFile('../audio_data/training/training-e','.wav')
    train_e_X=audioTool.cutAudio(4096)
    train_e_Y=audioTool.get_csv()
    audioTool.findFile('../audio_data/training/training-f','.wav')
    train_f_X=audioTool.cutAudio(4096)
    train_f_Y=audioTool.get_csv()
    train_X=union(union(union(train_a_X,train_b_X),union(train_c_X,train_d_X)),union(train_e_X,train_f_X))
    train_Y=union(union(union(train_a_Y,train_b_Y),union(train_c_Y,train_d_Y)),union(train_e_Y,train_f_Y))
    print(train_a_X.shape,train_a_Y.shape)
    print(train_b_X.shape,train_b_Y.shape)
    print(train_c_X.shape,train_c_Y.shape)
    print(train_d_X.shape,train_d_Y.shape)
    print(train_e_X.shape,train_e_Y.shape)
    print(train_f_X.shape,train_f_Y.shape)
    audioTool.findFile('../audio_data/validation','.wav')
    validation_X=audioTool.cutAudio(4096)
    validation_Y=audioTool.get_csv()

    print(train_X.shape)
    print(train_Y.shape)
    print(validation_X.shape)
    print(validation_Y.shape)
###
    train= ToHdf5()
    train.writeData('test_4096_more.h5',train_X,train_Y,validation_X,validation_Y)
#    train_X,train_Y,validation_X,validation_Y=train.loadData('test_4096.h5')
#    print(train_X.shape)
#    print(train_Y.shape)
#    print(validation_X.shape)
#    print(validation_Y.shape)
if __name__=='__main__':
    main()
