# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 18:12:43 2018

@author: shuan

this file is used to test other function, so ignore it!
"""
from pre_process import AudioTool
from pre_process import ToHdf5
import numpy as np
def main():
    print('model launch')
    audioTool=AudioTool('./audio_data/training-a/a0001.wav')
    segVector=audioTool.getSegment(length=2000,isPlot=True)
    print('截取长度:2000')
    
    tx=np.random.randn(20,2048,1)
    ty=np.random.randn(20,1)
    vx=np.random.randn(10,2048)
    vy=np.random.randn(10)
    train= ToHdf5()
    train.writeData('testHdf5.hdf5',tx,ty,vx,vy)
#    a,b,b,d=train.loadData('testHdf5.hdf5')
#    print(a[1,:,:])
if __name__=='__main__':
    main()
