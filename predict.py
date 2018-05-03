# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 15:26:24 2018

@author: shuan
"""
from model import heartSoundModel
from pre_process import ToHdf5
import numpy as np

train=ToHdf5()
X_train,Y_train,X_validation,Y_validation=train.loadData('test_4096_more.h5')
hmodel=heartSoundModel(X_train.shape[1:])
hmodel.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    
def train():
    hmodel.fit(x=X_train,y=Y_train,epochs=10,batch_size=200)
    #hmodel.train_on_batch(X_train,Y_train)
    hmodel.save_weights("weight_4096.h5")
    
def validation():
    hmodel.load_weights('weight_4096.h5')
    print(hmodel.evaluate(X_validation,Y_validation))    

def main():
    train()
    validation()
if __name__=='__main__':
    main()
    