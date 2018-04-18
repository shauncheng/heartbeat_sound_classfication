# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 18:12:43 2018

@author: shuan

this file is used to test other function, so ignore it!
"""
from pre_process import AudioTool
from pre_process import ToHdf5
import h5py
import numpy as np
def main():
    print('model launch')
#    audioTool=AudioTool('./audio_data/training-a','.wav')
#    train_X=audioTool.cutAudio(2048)
#    train_Y=audioTool.get_csv()
#    print(train_Y[1:10,:])
    train_X=np.random.randn(6326,2048,1)
    train_Y=np.random.randint(2,size=(6326,1))
    validation_X=np.zeros((2,2))
    validation_Y=np.zeros((2,2))
    print(train_X.shape)
    print(train_Y.shape)
    print(validation_X.shape)
    print(validation_Y.shape)
    

#    hf=h5py.File('test.h5','w')
#    hf.close()
#    train=hf.create_group("train")
#    vali=hf.create_group("validation")
#    train["train_X"]=train_X
#    train["train_Y"]=train_Y
#    vali["validation_X"]=validation_X
#    vali["validation_Y"]=validation_Y
#    print(hf["train"]["train_X"].value)
    
    
    train= ToHdf5()
    train.writeData('test.h5',train_X,train_Y,validation_X,validation_Y)
    a,b,b,d=train.loadData('test.h5')
    print(a.shape)
    print(b.shape)
if __name__=='__main__':
    main()
