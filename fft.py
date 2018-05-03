# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:28:37 2018

@author: shuan
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
    
    

def test():
    sampling_rate = 8000
    fft_size = 512
    t = np.arange(0, 1.0, 1.0/sampling_rate)
    x = np.sin(2*np.pi*156.25*t)  + 2*np.sin(2*np.pi*234.375*t)
    xs = x[:fft_size]
    xf = np.fft.rfft(xs)/fft_size
    freqs = np.linspace(0, sampling_rate/2, fft_size/2+1)
    xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    print(t.shape)
    print(t[:fft_size].shape)
    print(xs.shape)
    print(xf.shape)
    print(freqs.shape)
    print(xfp.shape)
    pl.figure(figsize=(8,4))
    pl.subplot(211)
    pl.plot(t[:fft_size], xs)
    pl.xlabel(u"时间(秒)")
    pl.title(u"156.25Hz和234.375Hz的波形和频谱")
    pl.subplot(212)
    pl.plot(freqs, xfp)
    pl.xlabel(u"频率(Hz)")
    pl.subplots_adjust(hspace=0.4)
    pl.show()


def main():
    av=readInfo('../training/training-a/a0001.wav')
    fft(av)
#    test()

if __name__=='__main__':
    main()