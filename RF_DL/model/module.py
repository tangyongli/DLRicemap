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
from tensorflow.keras.layers import Input, Reshape, LayerNormalization, multiply,Dense,Activation,Add,Flatten,Lambda ,Concatenate, Conv1D, Conv2D, Activation
from keras import backend as K
from keras.layers import Input, SeparableConv2D, Conv2D,DepthwiseConv2D
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
    # else:
    #     x = x
    return x

def ResBlock(inputs,channels,geoinputs,two=False,geodateattention=False,attentionbefore=False,attentionafter=False): # 11 7 7 11
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
        x=conbatchrelu_block(inputs,channels,3,1,1,False,False)
        if attentionbefore==True:
            x=channel_attention(x,geoinputs,ratio=8,geo=geodateattention)
            x=spatial_attention(x)
        if inputs.shape[-1]==x.shape[-1]:
            x = layers.Add()([inputs, x])  # 
        else:
            inputs=layers.Conv2D(x.shape[-1], (1,1), padding="same",kernel_initializer='he_normal', use_bias=False)(inputs)
            inputs = layers.BatchNormalization()(inputs)
            x = layers.Add()([inputs, x])
    else:
        x=conbatchrelu_block(inputs,channels,3,1,1,False,True)
        x=conbatchrelu_block(x,channels,3,1,1,False,False)
        if attentionbefore==True:
            x=channel_attention(x,geoinputs,ratio=8,geo=geodateattention)
            x=spatial_attention(x)
        if inputs.shape[-1]==x.shape[-1]:
            x = layers.Add()([inputs, x])  # 
        else:
            inputs=layers.Conv2D(x.shape[-1], (1,1), padding="same",kernel_initializer='he_normal', use_bias=False)(inputs)
            inputs= layers.BatchNormalization()(inputs)
            x = layers.Add()([inputs, x])
    x=layers.ReLU()(x)
    if attentionafter==True:
        x=channel_attention(x,geoinputs,ratio=8,geo=geodateattention)
        x=spatial_attention(x)
    return x

# import tensorflow as tf
# import tensorflow.contrib.slim as slim

def deepwise2d(inputs,k,ratio=8,cbrm=False,coord=False,name='one'):
    x1=layers.Conv2D(inputs.shape[-1]*k,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(inputs) 
    x1=layers.BatchNormalization()(x1)
    x1= layers.ReLU(max_value=6)(x1)
    x=DepthwiseConv2D(3,padding='same',activation='relu')(x1)
    # x=DepthwiseConv2D(3,padding='same',activation='relu')(x)
    # x=coordinate(x,ratio=8)
    x=layers.Conv2D(inputs.shape[-1],(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    x=Add()([inputs,x])
    print('dpwise',x.shape)
    if cbrm:
        x=channel_attention(x,ratio=ratio)
        x=spatial_attention(x)
    if coord:
        x=coordinate(x,ratio=32)
    # x=layers.concatenate([inputs,x],axis=-1) 
    return x
inputs=keras.Input((11,11,64))
# outputs=deepwise2d(inputs,k=6)
# model=keras.Model(inputs=inputs,outputs=outputs)
# model.summary()













def se_block(input_feature, ratio=8):
	"""Contains the implementation of Squeeze-and-Excitation(SE) block.
	As described in https://arxiv.org/abs/1709.01507.
	"""
	
	# channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature.shape[-1]#_keras_shape[channel_axis]

	se_feature = GlobalAveragePooling1D()(input_feature)
	se_feature = Reshape((1, 1, channel))(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel)
	se_feature = Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel//ratio)
	se_feature = Dense(channel,
					   activation='sigmoid',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel)
	if K.image_data_format() == 'channels_first':
		se_feature = Permute((3, 1, 2))(se_feature)

	se_feature = multiply([input_feature, se_feature])
	return se_feature




def CoordAtt(x, reduction = 32):

    def coord_act(x):
        tmpx = tf.nn.relu6(x+3) / 6
        x = x * tmpx
        return x

    x_shape = x.get_shape().as_list()
    [b, h, w, c] = x_shape
    x_h = slim.avg_pool2d(x, kernel_size = [1, w], stride = 1)
    x_w = slim.avg_pool2d(x, kernel_size = [h, 1], stride = 1)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])

    y = tf.concat([x_h, x_w], axis=1)
    mip = max(8, c // reduction)
    y = slim.conv2d(y, mip, (1, 1), stride=1, padding='VALID', normalizer_fn = slim.batch_norm, activation_fn=coord_act,scope='ca_conv1')

    x_h, x_w = tf.split(y, num_or_size_splits=2, axis=1)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])
    a_h = slim.conv2d(x_h, c, (1, 1), stride=1, padding='VALID', normalizer_fn = None, activation_fn=tf.nn.sigmoid,scope='ca_conv2')
    a_w = slim.conv2d(x_w, c, (1, 1), stride=1, padding='VALID', normalizer_fn = None, activation_fn=tf.nn.sigmoid,scope='ca_conv3')

    out = x * a_h * a_w


    return out

import tensorflow as tf
from keras.layers import Lambda,Concatenate,Reshape,Conv2D,BatchNormalization,Activation,Multiply,Add

def coordinate(inputs,ratio=32, name="name"):
    def coord_act(x):
        tmpx = tf.nn.relu6(x+3) / 6
        x = x * tmpx
        return x
    H,W,C = [int(x) for x in inputs.shape[1:]]
    # temp_dim = max(int(C//ratio),ratio)
    mip = max(8, C//ratio)
    H_pool = Lambda(lambda x: tf.reduce_mean(x, axis=1))(inputs)
    W_pool = Lambda(lambda x: tf.reduce_mean(x, axis=2))(inputs)
    x = Concatenate(axis=1)([H_pool,W_pool])
    x = Reshape((1,W+H,C))(x)
    x = Conv2D(mip,1)(x)
    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x=relu(x + 3) / 6
    x=coord_act(x)
    x_h,x_w = Lambda(lambda x:tf.split(x,[H,W],axis=2))(x)
    x_w = Reshape((W,1,mip))(x_w)

    x_h = Conv2D(C,1,activation='sigmoid')(x_h)
    x_w = Conv2D(C, 1, activation='sigmoid')(x_w)
    x = Multiply()([inputs,x_h,x_w])
    # x = Add()([inputs,x])
    return x














# class eca_layer():
#     """Constructs a ECA module.

#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#     def __init__(self, x, k_size=3):
#         super(eca_layer, self).__init__()
#         self.avg_pool = layers.GlobalAveragePooling2D()
#         self.conv = layers.Conv1D(1, 3, 1,padding='same', use_bias=False)  #128,(1,1),strides=(1,1),padding='same'
#         self.sigmoid = tf.keras.activations.sigmoid
#         self.x=x

#     def forward(self, x):
#         # feature descriptor on the global spatial information
#         y = self.avg_pool(self.x)

#         # Two different branches of ECA module
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

#         # Multi-scale information fusion
#         y = self.sigmoid(y)

#         return self.x * y.expand_as(self.x)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
import math
 
 
def eca_block(inputs, b=1, gama=2):
    # 输入特征图的通道数
    in_channel = inputs.shape[-1]
 
    # 根据公式计算自适应卷积核大小
    kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
 
    # 如果卷积核大小是偶数，就使用它
    if kernel_size % 2:
        kernel_size = kernel_size
 
    # 如果卷积核大小是奇数就变成偶数
    else:
        kernel_size = kernel_size + 1
 
    # [h,w,c]==>[None,c] 全局平均池化
    x = layers.GlobalAveragePooling2D()(inputs)
 
    # [None,c]==>[c,1]
    x = layers.Reshape(target_shape=(in_channel, 1))(x)
 
    # [c,1]==>[c,1]
    x = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)
 
    # sigmoid激活
    x = tf.nn.sigmoid(x)
 
    # [c,1]==>[1,1,c]
    x = layers.Reshape((1, 1, in_channel))(x)
 
    # 结果和输入相乘
    outputs = layers.multiply([inputs, x])
 
    return outputs


def channel_attention(inputs,ratio=8):
    
    # 通道维度上的平均池化
    # avg_pool= layers.TimeDistributed(layers.GlobalAveragePooling2D())(input_feature)
    avg_pool= layers.GlobalAveragePooling2D()(inputs)
    max_pool = layers.GlobalMaxPooling2D()(inputs)
    # print('11111111111111111111111111111',avg_pool.shape,geoinputs.shape)
    # if dense:
    #     avggeo_pool = Concatenate(axis=-1)([avg_pool, geoinputs])
    #     maxgeo_pool = Concatenate(axis=-1)([max_pool, geoinputs])
    #     channel=avggeo_pool.shape[-1]
    #     # geotimechannel=4
    #     shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    #     shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    #     avg_pool1=shared_layer_one(avggeo_pool)
    #     avg_pool1=shared_layer_two(avg_pool1)
    #     max_pool1=shared_layer_one(maxgeo_pool)
    #     max_pool1=shared_layer_two(max_pool1)

    #     dense=Add()([avg_pool1,max_pool1])
    #     dense = Lambda(lambda x: K.reshape(x, (K.shape(x)[0], 1, 1, K.shape(x)[-1])))(dense)
   
    # if geo:
    #     # avggeo_pool = layers.concatenate([avg_pool, geoinputs],axis=-1)
    #     # print('geotruetruetruetruetrue' ,avggeo_pool.shape)
    #     # maxgeo_pool = layers.concatenate([max_pool, geoinputs], axis=-1)
    #     avggeo_pool = Concatenate(axis=-1)([avg_pool, geoinputs])
    #     maxgeo_pool = Concatenate(axis=-1)([max_pool, geoinputs])
    #     geotimechannel=geoinputs.shape[-1]
    #     # avggeo_pool = avg_pool
    #     # maxgeo_pool=max_pool
    #     # geotimechannel=0
    # else:
    #     avggeo_pool = avg_pool
    #     maxgeo_pool=max_pool
    #     geotimechannel=0
   
    avg_pool=Reshape((1,1,avg_pool.shape[-1]))(avg_pool)
    max_pool = Reshape((1,1,max_pool.shape[-1]))(max_pool)
    # avg_pool = Lambda(lambda x: K.reshape(x, (K.shape(x)[0], 1, 1, K.shape(x)[-1])))(avggeo_pool)
    # max_pool= Lambda(lambda x: K.reshape(x, (K.shape(x)[0], 1, 1, K.shape(x)[-1])))(maxgeo_pool)
    # print('avg_pool',avg_pool.shape)
    channel =  avg_pool.shape[-1]  # 获取通道维度
    # print('channel',channel)
    # print('rtatio',ratio)
    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    avg_pool2=shared_layer_one(avg_pool)
    avg_pool2=shared_layer_two(avg_pool)
    # print('avg2', avg_pool2.shape) 
    # print( max_pool.shape)
 
    max_pool2 = shared_layer_one(max_pool)
    max_pool2= shared_layer_two(max_pool)
    # print('max', max_pool2.shape) 
   
    # 通道注意力的输出
    cbam_feature = Add()([avg_pool2,max_pool2])
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
    # print("spation",cbam_feature.shape) #(None, 7,7 1)
    return multiply([input_feature, cbam_feature])
