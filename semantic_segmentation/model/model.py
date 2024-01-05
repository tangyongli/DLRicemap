
from module import *
def singlebranch(inputs,geoinputs,channelratio,cnn1d=False):
   
    x=layers.Conv2D(32,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(inputs)   
    x=layers.Conv2D(32,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(inputs)  
    x = keras.layers.BatchNormalization()(x)
    x= keras.layers.ReLU()(x)
    x=ResBlock(x,channel=64,geoinputs=geoinputs,two=True,attentionbefore=False,attentionafter=True)
    x=ResBlock(x,channel=128,geoinputs=geoinputs,two=True,attentionbefore=False,attentionafter=True)
    # 参数量很大
    # x=ResBlock(x,channel=256,geoinputs=geoinputs,two=True,attentionbefore=False,attentionafter=True)
    if cnn1d==True:
        x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    x=layers.concatenate([inputs,x],axis=-1) 
    return x
def DualCnn2dGeotimeCbrm(inputshape,channelratio,dual=True,dropout=0.2,L2=0):
    '''
    inputshape: (2,patchsize,patchsize,15). last three channels are doy,lat,lon
    channelratio:the ratio of channel atttention dense layer after max and average pooling
    '''
    if L2>0:
        reg = tf.keras.regularizers.l2(l=L2)
    else: 
        reg = tf.keras.regularizers.l2(l=0.0)
    inputs= keras.Input(shape=inputshape) #(None, 2, 11, 11, 15)
    inputtime1bands=layers.Lambda(lambda x: x[...,0:1,:,:,0:12])(inputs)
    print('inputtime1bands',inputtime1bands.shape,inputtime1bands.shape[0]) #(None, 1, 11, 11, 12) None
    inputtime1bands=Reshape((11,11,12))(inputtime1bands) #(None,  11, 11, 12)
    inputtime2bands=layers.Lambda(lambda x: x[...,0:1,:,:,0:12])(inputs)
    inputtime2bands=Reshape((11,11,12))(inputtime2bands) #(None, 11, 11, 12)
    inputtime1geotimeinputs= layers.Lambda(lambda x: x[...,0:1,inputs.shape[2]//2:inputs.shape[2]//2+1,inputs.shape[3]//2:inputs.shape[3]//2+1,12:16])(inputs) # (1,1,3)
    inputtime2geotimeinputs= layers.Lambda(lambda x: x[...,1:2,inputs.shape[2]//2:inputs.shape[2]//2+1,inputs.shape[3]//2:inputs.shape[3]//2+1,12:16])(inputs) # (1,1,3)
    channel=inputtime1geotimeinputs.shape[-1]
    inputtime1geotimeinputs=Reshape((channel,))(inputtime1geotimeinputs) #(3,)
    inputtime2geotimeinputs=Reshape((channel,))(inputtime2geotimeinputs) #(3,)
    print(  inputtime2geotimeinputs.shape)
    time1cnn=singlebranch(inputtime1bands,inputtime1geotimeinputs,channelratio,cnn1d=True)
    time2cnn=singlebranch(inputtime2bands,inputtime2geotimeinputs,channelratio,cnn1d=True)
    x=layers.concatenate([time1cnn,time2cnn],axis=-1) #add()?
    x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    x=layers.Conv2D(128,(3,3),strides=(1,1),padding='same',dilation_rate=(2, 2),kernel_initializer='he_normal', use_bias=False)(x) 
    x = layers.MaxPooling2D(strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x= keras.layers.ReLU()(x)
    x=channel_attention(x,x,channelratio,False)
    x=spatial_attention(x)
    # x=layers.Conv2D(256,(3,3),strides=(1,1),padding='same',dilation_rate=(2, 2),kernel_initializer='he_normal', use_bias=False)(x) 
    # x = keras.layers.BatchNormalization()(x)
    # x = layers.MaxPooling2D(strides=2, padding="same")(x)
    # x= keras.layers.ReLU()(x)
    # x=channel_attention(x,x,channelratio,False)
    # x=spatial_attention(x)
    x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    # x=layers.AveragePooling2D(pool_size=3)(x)
    x=layers.GlobalAveragePooling2D()(x)
    x=Flatten()(x)
    # x=Dense(512,kernel_regularizer=reg)(x)
    x=Dense(256,kernel_regularizer=reg)(x)
    x= layers.Dropout(dropout)(x)
    print('x',x.shape)
    output_layer=Dense(2,kernel_regularizer=reg,activation='softmax')(x) #,activation='sigmoid'
    return Model(inputs,output_layer)
#%%
# inputshape=(2,11,11,16)
# numFilters=[16,32,64,128,256,512]
# geoinput=(3,)
# model=DualCnn2dGeotimeCbrm(inputshape,4,0,0.002,True) # ratio越小参数越大
# print(model)
# print(model.summary())
