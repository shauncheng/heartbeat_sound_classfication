# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 18:12:43 2018

@author: shuan
"""
from pre_process import AudioTool

def main():
    print('model launch')
    audioTool=AudioTool('./audio_data/training-a/a0001.wav')
    segVector=audioTool.getSegment(length=2000,isPlot=True)
    print('截取长度:2000')
    
    
if __name__=='__main__':
    main()
