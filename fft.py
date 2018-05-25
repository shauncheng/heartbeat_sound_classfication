# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:28:37 2018

@author: shuan
快速傅里叶变换
"""

import wave
import numpy as np
import pylab as pl  

(nchannels, sampwidth, framerate, nframes, comptype, compname)=(None,None,None,None,None,None)
def readInfo(path):
    wavefile = wave.open(path,'r')
    nchannels, sampwidth, framerate, nframes, comptype, compname=wavefile.getparams()
    audioVector = wavefile.readframes(nframes)
    audioVector = np.frombuffer(audioVector,dtype = np.short)
    return audioVector

def fft(vector):
    segvec=vector[:4096]
    pl.figure(figsize=(20,50))
    xf=np.fft.rfft(segvec)/4096
    pl.subplot(6,1,1)
    pl.plot(xf)
    xfp = np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    pl.subplot(6,1,2)
    pl.plot(xfp)
    
    segvec=vector[5000:9096]
    xf=np.fft.rfft(segvec)/4096
    pl.subplot(6,1,3)
    pl.plot(xf)
    xfp = np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    pl.subplot(6,1,4)
    pl.plot(xfp)
    
    segvec=vector[10000:14096]
    xf=np.fft.rfft(segvec)/4096
    pl.subplot(6,1,5)
    pl.plot(xf)
    xfp = np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    pl.subplot(6,1,6)
    pl.plot(xfp)
    
