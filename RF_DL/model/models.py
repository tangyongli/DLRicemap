
from RF_DL.model.module import *
from keras.layers import Input, SeparableConv2D, DepthwiseConv2D


def resnetattention(inputs,channelratio=8,cnn1d=False,sar=False):
    print('input',inputs.shape)
    x=layers.Conv2D(16,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(inputs)   
    x=layers.Conv2D(16,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)  
    print('none1',x.shape)
    x = keras.layers.BatchNormalization()(x)
    x= keras.layers.ReLU()(x)
    x=ResBlock(inputs,32,two=True,geodateattention=False,attentionbefore=False,attentionafter=False)
    x=layers.MaxPooling2D((2,2))(x)
    x=ResBlock(inputs,64,two=True,geodateattention=False,attentionbefore=False,attentionafter=False)
    x=layers.MaxPooling2D((2,2))(x)
    x=ResBlock(inputs,128,two=True,geodateattention=False,attentionbefore=False,attentionafter=False)
    x=layers.MaxPooling2D((2,2))(x)
    x=ResBlock(inputs,256,two=True,geodateattention=False,attentionbefore=False,attentionafter=False)
    x=layers.MaxPooling2D((2,2))(x)
    x=layers.Conv2D(256,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    if sar==False:
        x=deepwise2d(x,k=6,ratio=channelratio,cbrm=True,name='three')
        x=layers.concatenate([inputs,x],axis=-1) 
        x=layers.Conv2D(128,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
        x5=deepwise2d(x,k=6,ratio=channelratio,cbrm=True,name='four')
        x=layers.concatenate([inputs,x5],axis=-1) 

    if cnn1d==True:
        x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    x=eca_block(x, b=1, gama=2)
    x=GlobalAveragePooling2D()(x)
    x=Flatten()(x)
 
    return x
def depthwiseattention(inputs,channelratio=8,depthwisecbrm=False,depthwisecorrd=False):
    print('input',inputs.shape)
  
    x=layers.Conv2D(16,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(inputs)   
    x=layers.Conv2D(16,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)  
    print('none1',x.shape)
    x = keras.layers.BatchNormalization()(x)
    x1= keras.layers.ReLU()(x)
    print('none2',x.shape)
    x=deepwise2d(x1,k=6,ratio=channelratio,cbrm=False,coord=False,name='zero') 
    x=layers.concatenate([inputs,x],axis=-1) 
    x2=layers.Conv2D(32,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)  
    # x2=layers.MaxPooling2D((2,2))(x)
    x=deepwise2d(x2,k=6,ratio=channelratio,cbrm=False,coord=False,name='one')
    x=layers.concatenate([inputs,x],axis=-1) 
    x3=layers.Conv2D(64,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)  
    # x3=layers.MaxPooling2D((2,2))(x)
    x=deepwise2d(x3,k=6,ratio=channelratio,cbrm=depthwisecbrm,coord=depthwisecorrd,name='two')
    x=layers.concatenate([inputs,x],axis=-1) 
    x=layers.Conv2D(96,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    # x4=layers.MaxPooling2D((2,2))(x)
    x=deepwise2d(x,k=6,ratio=channelratio,cbrm=depthwisecbrm,coord=depthwisecorrd,name='three')
    x=layers.concatenate([inputs,x],axis=-1) 
    x=layers.Conv2D(128,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    x5=deepwise2d(x,k=6,ratio=channelratio,cbrm=depthwisecbrm,coord=depthwisecorrd,name='four')
    x=layers.concatenate([inputs,x5],axis=-1) 
    # if cnn1d==True:
    #     # x=Reshape((x.shape[1],x.shape[3],x.shape[2]))(x)
        
    #     x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    # x=eca_block(x, b=1, gama=2)
    # x=GlobalAveragePooling2D()(x)
    # x=Flatten()(x)
    # x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    return x
def simplecnn2d(inputshape,num_filters,dropratio):
    inputs= keras.Input(shape=inputshape) # 11,11,121
    x=layers.Conv2D(num_filters[0], 3, strides=1,padding="same", activation='relu')(inputs)
    x=layers.Conv2D(num_filters[1],3,strides=1,padding='same', activation='relu')(x) # 
   
    # x=ResBlock(x,num_filters[1]) 
    x=layers.Conv2D(num_filters[2],3,strides=1,padding='same', activation='relu')(x) #
    x=layers.Conv2D(num_filters[3],3,strides=1,padding='same', activation='relu')(x) # 
    x=layers.MaxPooling2D()(x)
    x=layers.Conv2D(num_filters[4],3,strides=1,padding='same', activation='relu')(x) 
    x=GlobalAveragePooling2D()(x)
    x=Dense(512)(x)
    x= layers.Dropout(dropratio)(x)
    output_layer=Dense(2,activation='softmax')(x)
    return Model(inputs, output_layer)

def m2(inputs,channelratio,onlybands=True,dropout=0,L2=0):
    # Define your model architecture here...
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
    output_layer=Dense(2,kernel_regularizer=reg,activation='softmax')(x) 
    

    return Model(inputs,output_layer)
'''
    inputshape: (patchsize,patchsize,bandsize). last three channels are doy,lat,lon
    channelratio:the ratio of channel atttention dense layer after max and average pooling
'''
# def medianCnn2d(inputshape,channelratio, inputtag,depthwisecbrm=True,depthwisecorrd=False,ecabeforedense=False,cbrmbeforedense=False,concat=False,dropout=0.2,L2=0):
 
#     if inputtag==0:
#         startbands,endbands=0,39
#     if  inputtag==1:
#         startbands,endbands=0,13
#     if  inputtag==2:
#         startbands,endbands=13,26
#     if  inputtag==3:
#          startbands,endbands=26,39
#     if L2>0:
#         reg = tf.keras.regularizers.l2(l=L2)
#     else: 
#         reg = tf.keras.regularizers.l2(l=0.0)
#     inputs= keras.Input(shape=inputshape) #(None, 2, 11, 11, 15)
    
#     x1=layers.Lambda(lambda x: x[...,startbands:endbands])(inputs)
#     inputheight,inputwidth,inputschannels=x1.shape[1],x1.shape[2],x1.shape[-1]
#     x=depthwiseattention(x1,channelratio=channelratio,depthwisecbrm=depthwisecbrm,depthwisecorrd=depthwisecorrd)
#     # x=depthwiseattention(x1,channelratio,cnn1d=True)
#     # x=layers.Conv2D(128,(3,3),strides=(2,2),padding='same',dilation_rate=(1, 1),kernel_initializer='he_normal', use_bias=False)(x)
#     # x=Reshape((x.shape[1],x.shape[3],x.shape[2]))(x)
    
#     x=layers.Conv2D(256,(3,3),strides=(2,2),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
#     # x=layers.Conv2D(256,(3,3),strides=(1,1),padding='same',dilation_rate=(2, 2),kernel_initializer='he_normal', use_bias=False)(x) 
#     x = keras.layers.BatchNormalization()(x)
#     x= keras.layers.ReLU()(x)
#     x = layers.MaxPooling2D(strides=2, padding="same")(x)
   
#     x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
#     if ecabeforedense==True:
#          x=eca_block(x, b=1, gama=2)
#     if cbrmbeforedense==True:
#         x=channel_attention(x, 8) #inputs,geoinputs,ratio=8,geo=True,dense=False
#         x=spatial_attention(x)
#     x=layers.GlobalAveragePooling2D()(x)
#     x=Flatten()(x)
#     x=Dense(256,kernel_regularizer=reg)(x)
#     if concat==True:
#         xcenter=layers.Lambda(lambda x: x[...,inputheight//2:inputheight//2+1,inputwidth//2:inputwidth//2+1,0:inputschannels])(x1)
#         # xcenteravg=layers.GlobalAveragePooling2D()(inputs)
#         # xcentermax=layers.GlobalMaxPooling2D()(inputs)
#         # xcenter=layers.add([xcenteravg,xcentermax])
#         xcenter=Reshape((xcenter.shape[-1],))(xcenter)
#         xcenter=Dense(256,kernel_regularizer=reg)(xcenter)
#     # 如果是增加的话，加权平均？dense层能否做通道注意力？输入层的信息到密集层是直接通过dense还是conv1d?
#         x=layers.concatenate([xcenter,x],axis=-1)
#     x= layers.Dropout(dropout)(x)
#     x=Dense(64,kernel_regularizer=reg)(x)
#     x= layers.Dropout(dropout)(x)
#     print('x',x.shape)
#     output_layer=Dense(2,kernel_regularizer=reg,activation='softmax')(x) #,activation='sigmoid'
#     # output_layer=Dense(1, activation='sigmoid')(x)

#     return Model(inputs,output_layer)

# model=medianCnn2d(inputshape=(11,11,39),channelratio=8,inputtag=0,concat=True,dropout=0,L2=0)
# model.summary()

















def DualCnn2dGeotimeCbrm(inputshape,channelratio,dual=True,bandswithindex=True,dropout=0.2,L2=0):
    '''
    inputshape: (2,patchsize,patchsize,15). last three channels are doy,lat,lon
    channelratio:the ratio of channel atttention dense layer after max and average pooling
    '''
    if bandswithindex:
        bands=17
    else:
        bands=9

    if L2>0:
        reg = tf.keras.regularizers.l2(l=L2)
    else: 
        reg = tf.keras.regularizers.l2(l=0.0)
    inputs= keras.Input(shape=inputshape) #(None, 2, 11, 11, 15)
    inputtime1bands=layers.Lambda(lambda x: x[...,0:1,:,:,0:bands])(inputs)
    print('inputs',  inputs.shape)
    print('inputtime1bands',inputtime1bands.shape,inputtime1bands.shape[0]) #(None, 1, 11, 11, 12) None
   

    inputtime1bands=Reshape((inputheight,inputwidth,bands))(inputtime1bands) #(None,  11, 11, 12)
    inputtime2bands=layers.Lambda(lambda x: x[...,0:1,:,:,0:bands])(inputs)
    inputtime2bands=Reshape((inputheight,inputwidth,bands))(inputtime2bands) #(None, 11, 11, 12)
    inputtime1geotimeinputs= layers.Lambda(lambda x: x[...,0:1,inputheight//2:inputheight//2+1,inputwidth//2:inputwidth//2+1,bands:bands+geotimebands])(inputs) # (1,1,3)
    inputtime2geotimeinputs= layers.Lambda(lambda x: x[...,1:2,inputheight//2:inputheight//2+1,inputwidth//2:inputwidth//2+1,bands:bands+geotimebands])(inputs) # (1,1,3)
    
    '''
    it's wrong. 
    ①TypeError: Unable to serialize KerasTensor(type_spec=TensorSpec(shape=(None, 2, 11, 11, 16), dtype=tf.float32, name='input_1'), name='input_1', description="created by layer 'input_1'") to JSON. Unrecognized type <class 'keras.engine.keras_tensor.KerasTensor'>.
    RecursionError: maximum recursion depth exceeded while calling a Python object
    ②inputtime1geotimeinputs= layers.Lambda(lambda x: x[...,0:1,inputs.shape[2]//2:inputs.shape[2]//2+1,inputs.shape[3]//2:inputs.shape[3]//2+1,12:16])(inputs) # (1,1,3)
    inputtime2geotimeinputs= layers.Lambda(lambda x: x[...,1:2,inputs.shape[2]//2:inputs.shape[2]//2+1,inputs.shape[3]//2:inputs.shape[3]//2+1,12:16])(inputs) # (1,1,3)
    In this Lambda layer, inputs.shape[2], inputs.shape[3], etc., 
    are dynamically determined based on the shape of inputs during runtime. 
    However, when saving a model to JSON, the serialization process requires static information about the model architecture,
    and dynamic properties like these might pose challenges during serialization.
    '''
   
    # inputtime1geotimeinputs = layers.Lambda(lambda x: x[..., 0:1, K.int_shape(x)[2] // 2:K.int_shape(x)[2] // 2 + 1, K.int_shape(x)[3] // 2:K.int_shape(x)[3] // 2 + 1, 12:16])(inputs)
    # inputtime2geotimeinputs = layers.Lambda(lambda x: x[..., 1:2, K.int_shape(x)[2] // 2:K.int_shape(x)[2] // 2 + 1, K.int_shape(x)[3] // 2:K.int_shape(x)[3] // 2 + 1, 12:16])(inputs)
    channel=inputtime1geotimeinputs.shape[-1]
    inputtime1geotimeinputs=Reshape((channel,))(inputtime1geotimeinputs) #(3,)
    inputtime2geotimeinputs=Reshape((channel,))(inputtime2geotimeinputs) #(3,)
   
    print(  inputtime2geotimeinputs.shape)
    time1cnn=singlebranch(inputtime1bands,inputtime1geotimeinputs,channelratio,geo=True,cnn1d=True)
   
    time2cnn=singlebranch(inputtime2bands,inputtime2geotimeinputs,channelratio,geo=True,cnn1d=True)
    x=layers.concatenate([time2cnn,time1cnn],axis=-1) #add()?
    # x=eca_block(x, b=1, gama=2)
    x=layers.Conv2D(128,(3,3),strides=(2,2),padding='same',dilation_rate=(1, 1),kernel_initializer='he_normal', use_bias=False)(x)
    # x=Reshape((x.shape[1],x.shape[3],x.shape[2]))(x)
    x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    # x=layers.Conv2D(256,(3,3),strides=(2,2),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    # x=layers.Conv2D(256,(3,3),strides=(2,2),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    # x = layers.MaxPooling2D(strides=2, padding="same")(x)
    # x = keras.layers.BatchNormalization()(x)
    # x= keras.layers.ReLU()(x)
    # x=channel_attention(x,,channelratio,False)
    # x=spatial_attention(x)
    # x=layers.Conv2D(256,(3,3),strides=(1,1),padding='same',dilation_rate=(2, 2),kernel_initializer='he_normal', use_bias=False)(x) 
    # x = keras.layers.BatchNormalization()(x)
    # x = layers.MaxPooling2D(strides=2, padding="same")(x)
    # x= keras.layers.ReLU()(x)
    x=channel_attention(x,inputtime1geotimeinputs,8,False,False) #inputs,geoinputs,ratio=8,geo=True,dense=False
    x=spatial_attention(x)
    
    # x=layers.Conv1D(64,(3,),strides=(2,),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    # x=layers.AveragePooling2D(pool_size=3)(x)
    # x=eca_block(x, b=1, gama=2)


    # x=layers.GlobalAveragePooling2D()(x)
 
    # x=Dense(512,kernel_regularizer=reg)(x)
    
    x=layers.GlobalAveragePooling2D()(x)
    x=Flatten()(x)
    x=Dense(512,kernel_regularizer=reg)(x)
    x= layers.Dropout(dropout)(x)
    x=Dense(256,kernel_regularizer=reg)(x)
    x= layers.Dropout(dropout)(x)
    print('x',x.shape)
    output_layer=Dense(2,kernel_regularizer=reg,activation='softmax')(x) #,activation='sigmoid'
    # output_layer=Dense(1, activation='sigmoid')(x)

    return Model(inputs,output_layer)
