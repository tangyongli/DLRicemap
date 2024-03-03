

import numpy as np
import time
import random
import tensorflow as tf
from keras import backend as K
from keras import callbacks
# from model.models import DualCnn2dGeotimeCbrm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from cfgs import *
from semantic_segmentation.dataset.transformer_ImageDataGenerator import dataagument,val_data_generator#augment_data
from semantic_segmentation.dataset import transformer
from semantic_segmentation.dataset.transformer import  Compose,RandomRotation,RandomContrast,RandomScale,RandomHorizontalFlip,RandomVerticalFlip
from keras.utils import to_categorical
from RF_DL.model.models import *
# from model.models import singlebranch
from keras.layers import Input, SeparableConv2D, DepthwiseConv2D, GlobalAveragePooling1D
from RF_DL.plot import *




def medianCnn2d1(inputshape,channelratio,onlybands=True,dropout=0.2,L2=0):
    '''
    inputshape: (patchsize,patchsize,25). last three channels are doy,lat,lon
    channelratio:the ratio of channel atttention dense layer after max and average pooling
    '''
    if onlybands:
        bands=9
    else:
        bands=25

    if L2>0:
        reg = tf.keras.regularizers.l2(l=L2)
    else: 
        reg = tf.keras.regularizers.l2(l=0.0)
    inputs= keras.Input(shape=inputshape) #(None, 2, 11, 11, 15)
    # print('inputs',  inputs.shape)
    xbands=layers.Lambda(lambda x: x[...,0:17])(inputs)
    xglcm=layers.Lambda(lambda x: x[...,19:35])(inputs)
    x=layers.concatenate([xbands,xglcm],axis=-1)
    # xcenter=layers.Lambda(lambda x: x[...,5:6,5:6,0:9])(inputs)
    # xbands=inputs
    # xcenteravg=layers.GlobalAveragePooling2D()(xbands)
    # xcentermax=layers.GlobalMaxPooling2D()(xbands)
    # xcenter=layers.add([xcenteravg,xcentermax])
    # xcenter=Reshape((xcenter.shape[-1],))(xcenter)
    # xglcm=layers.Lambda(lambda x: x[...,9:17])(inputs)
    # xglcm=layers.Conv2D(8,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(xglcm)
    x=xbands#layers.concatenate([xbands,xglcm],axis=-1)
    x=depthwiseattention(x,channelratio,cnn1d=False,sar=False)
    x=layers.Conv2D(128,(3,3),strides=(2,2),padding='same',dilation_rate=(1, 1),kernel_initializer='he_normal', use_bias=False)(x)
    # x=Reshape((x.shape[1],x.shape[3],x.shape[2]))(x)
    
    x=layers.Conv2D(256,(3,3),strides=(2,2),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    # x=layers.Conv2D(256,(3,3),strides=(1,1),padding='same',dilation_rate=(2, 2),kernel_initializer='he_normal', use_bias=False)(x) 
    x = keras.layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(strides=2, padding="same")(x)
    x= keras.layers.ReLU()(x)
    x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    x=channel_attention(x, 8) #inputs,geoinputs,ratio=8,geo=True,dense=False
    x=spatial_attention(x)
    x=layers.GlobalAveragePooling2D()(x)
    x=Flatten()(x)
    # x=Dense(1024,kernel_regularizer=reg)(x)
    # x= layers.Dropout(dropout)(x)
    x=Dense(256,kernel_regularizer=reg)(x)
    x= layers.Dropout(dropout)(x)
    x=Dense(64,kernel_regularizer=reg)(x)
    x= layers.Dropout(dropout)(x)
    # xcenter=Dense(256,kernel_regularizer=reg)(xcenter)
    # # 如果是增加的话，加权平均？dense层能否做通道注意力？输入层的信息到密集层是直接通过dense还是conv1d?
    # xdense=layers.concatenate([xcenter,x],axis=-1)
    output_layer=Dense(2,kernel_regularizer=reg,activation='softmax')(x) #,activation='sigmoid'
        # output_layer=Dense(1, activation='sigmoid')(x)

    return Model(inputs,output_layer)