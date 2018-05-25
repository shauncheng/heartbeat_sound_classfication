# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 15:26:24 2018

@author: shuan
训练模型、验证模型表现
"""
from model import *
from pre_process import ToHdf5
import numpy as np

train=ToHdf5()
X_train,Y_train,X_validation,Y_validation=train.loadData('test_4096_more.h5')
hmodel=heartSoundModel_drop(X_train.shape[1:])
hmodel.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    
def train():
    hmodel.fit(x=X_train,y=Y_train,epochs=20,batch_size=200)
#    hmodel.train_on_batch(X_train,Y_train)
    hmodel.save_weights("weight_4096_more_dropoutreg.h5")
    6
def validation():
    hmodel.load_weights('weight_4096_more_dropoutreg.h5')
    print(hmodel.evaluate(X_validation,Y_validation))    
    pre=hmodel.predict(X_validation);
    pre=np.where(pre<0.5,0,1)
    TP=pre&Y_validation
    print(pre.shape)
    tp=np.count_nonzero(TP)
    fp=np.count_nonzero(pre)-tp
    fn=np.count_nonzero(Y_validation)-tp
    tn=pre.shape[0]-np.count_nonzero(pre)-fn
    accu=(tp+tn)/(tp+tn+fp+fn)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    print(tp,fp,fn,tn)
    print(accu,precision,recall)

def main():
    
#    train()
    validation()
    pass
    
if __name__=='__main__':
    main()
    