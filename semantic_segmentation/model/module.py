#%%
from tensorflow.keras import layers
import keras
from tensorflow.keras.layers import Input, Reshape, LayerNormalization, multiply,Dense,Activation,Add,Flatten,Lambda ,Concatenate, Conv1D, Conv2D
from keras import backend as K
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, BatchNormalization, Conv1D, Conv2D, Activation
from tensorflow.keras.layers import Dense, Softmax, Flatten,Lambda,Concatenate
# from models import cnn3dattention 
from tensorflow import expand_dims
from keras.models import Model
from keras.layers import GlobalAveragePooling2D,Dense
from tensorflow.keras import layers
from keras.utils import to_categorical
from tensorflow.keras.layers import Input, Reshape, LayerNormalization, multiply,Dense,Activation,Add,Flatten,Lambda ,Concatenate, Conv1D, Conv2D
from keras import backend as K
import math
import time
import os
#%%
def conbatchrelu_block(
    block_input,
    num_filters,
    kernel_size=3,
    strides=1,
    dilation_rate=(1,1),
    # padding="same",
    use_bias=False,use_refu=True
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    if use_refu==True:
        x = layers.ReLU()(x)
    else:
        x = x
    return x

def channel_attention(inputs,geoinputs,ratio=8,geo=True):
    
    # 通道维度上的平均池化
    # avg_pool= layers.TimeDistributed(layers.GlobalAveragePooling2D())(input_feature)
    avg_pool= layers.GlobalAveragePooling2D()(inputs)
    max_pool = layers.GlobalMaxPooling2D()(inputs)
    if geo==True:
        avggeo_pool = layers.concatenate([avg_pool, geoinputs],axis=-1)
        maxgeo_pool = layers.concatenate([max_pool, geoinputs], axis=-1)
        geotimechannel=geoinputs.shape[-1]
    else:
        avggeo_pool = avg_pool
        maxgeo_pool=max_pool
        geotimechannel=0

    avg_pool=Reshape((1,1,avggeo_pool.shape[-1]))(avggeo_pool)
    channel =  avggeo_pool.shape[-1]  # 获取通道维度
    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel-geotimechannel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    avg_pool=shared_layer_one(avg_pool)
    avg_pool=shared_layer_two(avg_pool)
    print('avg2', avg_pool.shape) 
    
    max_pool = Reshape((1,1,maxgeo_pool.shape[-1]))(maxgeo_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    print('max', max_pool.shape) 
    # 通道注意力的输出
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    return multiply([inputs, cbam_feature])
def spatial_attention(input_feature):
    # 空间注意力的计算，这里可以根据需要修改
    kernel_size = 5
    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input_feature)
    print(avg_pool.shape,max_pool.shape) #(None, 256, 256, 1) (None, 256, 256, 1)
    concat = Concatenate(axis=-1)([avg_pool, max_pool]) #(None, 256, 256, 1) (None, 256, 256, 2)

    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    print("spation",cbam_feature.shape) #(None, 7,7 1)
    return multiply([input_feature, cbam_feature])
#%%
def ResBlock(inputs,channel,geoinputs,two=False,attentionbefore=False,attentionafter=False): # 11 7 7 11
    '''
    block_input,
    num_filters,
    kernel_size=3,
    strides=1,
    dilation_rate=1,
    # padding="same",
    use_bias=False,use_refu=True
    '''
    if two==False:
        x=conbatchrelu_block(inputs,channel,3,1,1,False,False)
        if attentionbefore==True:
            x=channel_attention(x,geoinputs,ratio=8,geo=True)
            x=spatial_attention(x)
        if inputs.shape[-1]==x.shape[-1]:
            x = layers.Add()([inputs, x])  # 
        else:
            inputs=layers.Conv2D(x.shape[-1], (1,1), padding="same",kernel_initializer='he_normal', use_bias=False)(inputs)
            inputs = layers.BatchNormalization()(inputs)
            x = layers.Add()([inputs, x])
    else:
        x=conbatchrelu_block(inputs,channel,3,1,1,False,True)
        x=conbatchrelu_block(x,channel,3,1,1,False,False)
        if attentionbefore==True:
            x=channel_attention(x,geoinputs,ratio=8,geo=True)
            x=spatial_attention(x)
        if inputs.shape[-1]==x.shape[-1]:
            x = layers.Add()([inputs, x])  # 
        else:
            inputs=layers.Conv2D(x.shape[-1], (1,1), padding="same",kernel_initializer='he_normal', use_bias=False)(inputs)
            inputs= layers.BatchNormalization()(inputs)
            x = layers.Add()([inputs, x])
    x=layers.ReLU()(x)
    if attentionafter==True:
        x=channel_attention(x,geoinputs,ratio=8,geo=True)
        x=spatial_attention(x)
    return x
#%%

# %%
