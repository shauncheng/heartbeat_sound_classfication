# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 15:26:45 2018

@author: shuan
"""
import pylab as pl  
import wave
import numpy as np
import h5py
import os
import csv

##音频预处理
class AudioTool:
    wavefile=None
    fileList=[]
    cutNums=[]
    filePath=None
    (nchannels, sampwidth, framerate, nframes, comptype, compname)=(None,None,None,None,None,None)

#获取不同类型文件名存入fnames数组
    def __init__(self,path,fileType):
        self.filePath=path
        dirs = os.listdir(path)
        for i in dirs:
            if os.path.splitext(i)[1] == fileType:
                self.fileList.append(i)
        
#        self.wavefile=wave.open(path,'r')
#        (self.nchannels, self.sampwidth, self.framerate, self.nframes, self.comptype, self.compname)=self.wavefile.getparams()
#获取音频基本信息 输出（返回该段音频基本信息）   
#    def getAudioInfo(self):
#        return (self.nchannels, self.sampwidth, self.framerate, self.nframes, self.comptype, self.compname)

#音频截取，从多个周期当中截取一个周期以上长度，
    def cutAudio(self,cutLen):
        cutAudios = np.empty((0,cutLen))
        for i in range(len(self.fileList)):
            
#获取音频基本信息
            filepath=self.filePath
            Path = filepath+'/'+self.fileList[i]
            self.wavefile = wave.open(Path,'r')
            (self.nchannels, self.sampwidth, self.framerate, self.nframes, self.comptype, self.compname)=self.wavefile.getparams()
            
#获取音频文件向量信息
            audioVector = self.wavefile.readframes(self.nframes)
            audioVector = np.frombuffer(audioVector,dtype = np.short)
            
#裁剪单个音频存入segVectors矩阵            
            cutnum = self.nframes//cutLen//2
            self.cutNums.append(cutnum)
            segVectors = np.empty((cutnum,cutLen))
            segPos = 0
            for j in range(cutnum):
                segPos = np.random.randint(segPos,cutLen+segPos)
                segVector = audioVector[segPos:segPos+cutLen]
                segPos = segPos+cutLen
                segVectors[j,:] = segVector

#将所有截取到的音频存入cutAudios矩阵
            cutAudios = np.row_stack((cutAudios,segVectors))
        cutAudios = cutAudios.reshape(cutAudios.shape[0],cutLen,1)
        return cutAudios
 
#获取csv文件信息
    def get_csv(self):
        Path = self.filePath+"//REFERENCE.csv"
        csvfile = csv.reader(open(Path,'r'))
        inforY = []
        for i in csvfile:
            inforY.append(i[1])

#生成trainY矩阵            
        trainY = []
        for i in range(len(self.cutNums)):
            for j in range(self.cutNums[i]):
                trainY.append(inforY[i])
        trainY = np.array(trainY)
        trainY = trainY.reshape(trainY.shape[0],1)
        return trainY
        


#音频去噪处理
    def filterNoise():
        pass
    

#将处理过后的每一个训练集，向量存成hdf5文件
class ToHdf5:
    __fw=None
    __fr=None
    train=None
    validation=None

#将内存中数据写入hdf5文件中，输入（文件名，训练样本，训练标签，验证样本，验证标签）
    def writeData(self, newFile, train_X, train_Y, validation_X, validation_Y):
       self.fw=h5py.File(newFile,'w')
       self.train=self.fw.create_group("train")
       self.validation=self.fw.create_group("validation")
       self.train["train_X"]=train_X
       self.train["train_Y"]=train_Y
       self.validation["validation_X"]=validation_X
       self.validation["validation_Y"]=validation_Y
       self.fw.close()

#加载函数，将hdf5的数据加载到内存，输入（加载文件名）
    def loadData(self,name):
        self.fr=h5py.File("testHdf5.hdf5",'r')
        train_X=self.fr["train"]["train_X"].value
        train_Y=self.fr["train"]["train_Y"].value
        validation_X=self.fr["validation"]["validation_X"].value
        validation_Y=self.fr["validation"]["validation_Y"].value
        self.fr.close()
        return [train_X,train_Y,validation_X,validation_Y]
        
#测试函数，查看hdf5文件内容
    def visualization(self):
        print('visualzation')
        self.fr=h5py.File('test1.hdf5','r')
        for k in self.fr.keys():
            for i in self.fr[k].keys():
                print(self.fr[k][i])
        self.fr.close()