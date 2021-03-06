# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 15:26:45 2018

@author: shuan
"""

import wave
import numpy as np
import h5py
import os
import csv




##音频预处理
class AudioTool:
    """
    
    音频处理
    """
    wavefile=None
    fileList=[]
    cutNums=[]
    filePath=None
    (nchannels, sampwidth, framerate, nframes, comptype, compname)=(None,None,None,None,None,None)

#获取不同类型文件名存入fnames数组
    def findFile(self,path,fileType):
        self.fileList=[]
        self.filePath=path
        dirs = os.listdir(path)
        for i in dirs:
            if os.path.splitext(i)[1] == fileType:
                self.fileList.append(i)
        

#音频截取，从多个周期当中截取一个周期以上长度，
    def cutAudio(self,cutLen):
        self.cutNums=[]
        cutAudios = np.empty((0,int(cutLen/2)))
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
            cutnum = self.nframes//cutLen-1
            self.cutNums.append(cutnum)
            segVectors = np.empty((cutnum,int(cutLen/2)))
            segPos = 0
            for j in range(cutnum):
                segPos = np.random.randint(segPos,cutLen+segPos)
                segVector = audioVector[segPos:segPos+cutLen]
                if segVector.shape[0]<cutLen:
                    break
                segPos = segPos+1000
                segVectors[j,:] = self.fft(segVector,cutLen)

#归一化处理,将所有截取到的音频存入cutAudios矩阵
            cutAudios = np.row_stack((cutAudios,segVectors))
        cutAudios = cutAudios.reshape(cutAudios.shape[0],int(cutLen/2),1)
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
                inforY[i] = int(inforY[i])
                trainY.append(inforY[i])
        
        trainY = np.array(trainY)
        
        trainY = trainY.reshape(trainY.shape[0],1)
        return trainY
    
    def Resample(self,input_signal,src_fs,tar_fs):
        '''
    
        :param input_signal:输入信号
        :param src_fs:输入信号采样率
        :param tar_fs:输出信号采样率
        :return:输出信号
        '''
    
        dtype = input_signal.dtype
        audio_len = len(input_signal)
        audio_time_max = 1.0*(audio_len-1) / src_fs
        src_time = 1.0 * np.linspace(0,audio_len,audio_len) / src_fs
        tar_time = 1.0 * np.linspace(0,np.int(audio_time_max*tar_fs),np.int(audio_time_max*tar_fs)) / tar_fs
        output_signal = np.interp(tar_time,src_time,input_signal).astype(dtype)
    
        return output_signal

        


#音频去噪处理
    def fft(self,vector,cutLen):
        xf=np.fft.rfft(vector)/cutLen
        xfp = np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
        return xfp[:2048]
    

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
       
       
    def counDdata(self,name):
        data = self.loadData(name)
        train_Y = list(data[1].reshape(len(data[1]),))
        train_Y0 = train_Y.count(0)
        train_Y1 = train_Y.count(1)
        validation_Y = list(data[3].reshape(len(data[3]),))
        validation_Y0 = validation_Y.count(0)
        validation_Y1 = validation_Y.count(1)
        return [train_Y0,train_Y1,validation_Y0,validation_Y1]
        
#加载函数，将hdf5的数据加载到内存，输入（加载文件名）
    def loadData(self,name):
        self.fr=h5py.File(name,'r')
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
