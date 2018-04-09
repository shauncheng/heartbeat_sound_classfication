# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 15:25:22 2018

@author: shuan
"""
import tensorflow as tf
import numpy as np

#前向传播过程，输入（训练样本，参数）
def forword_propagate(X,parameters):
    pass

#计算代价值，输入（输出层线性值，Z=np.dot(W,X)）
def compute_cost(Z,Y):
    pass

#训练模型，输入（训练样本，训练标签，学习速率，更新代数，每一批迭代数量，是否可视化）
def model(X_train,Y_train,learning_rate=0.09,num_epochs=100,minibatch_size=64,print_cost=True):
    pass


