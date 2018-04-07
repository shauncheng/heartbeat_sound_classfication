# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 15:26:45 2018

@author: shuan
"""
import pylab as pl  
import wave
import numpy as np
class AudioTool:
    wavefile=None
    (nchannels, sampwidth, framerate, nframes, comptype, compname)=(None,None,None,None,None,None)
    def __init__(self,path):
        self.wavefile=wave.open(path,'r')
        (self.nchannels, self.sampwidth, self.framerate, self.nframes, self.comptype, self.compname)=self.wavefile.getparams()
    
#获取音频基本信息    
    def getAudioInfo(self):
        return (self.nchannels, self.sampwidth, self.framerate, self.nframes, self.comptype, self.compname)

#音频截取，从多个周期当中截取一个周期以上长度，
#输入截取长度，是否可视化
#输出截取向量
    def getSegment(self, length, isPlot=False):
        audioVector=self.wavefile.readframes(self.nframes)
        audioVector = np.frombuffer(audioVector,dtype = np.short)
        segPos=np.random.randint(0,self.nframes-length-1000)
        print('截取位置:',segPos)
        segVector=audioVector[segPos:segPos+length]
        if isPlot:
            newVector=np.zeros(self.nframes)
            newVector[segPos:segPos+length]=segVector
            pl.plot(audioVector/6400)
            pl.plot(newVector/6400)
        return segVector

#音频去噪处理
    def filterNoise():
        pass