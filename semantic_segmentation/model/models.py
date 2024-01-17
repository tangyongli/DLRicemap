
from semantic_segmentation.model.module import *
from keras.layers import Input, SeparableConv2D, DepthwiseConv2D
from cfgs import *
# from semantic_segmentation.dataset.datagenerate import get_c_value
# print(get_c_value())
def singlebranch(inputs,channelratio=8,cnn1d=False,sar=False):
  
    x=layers.Conv2D(16,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(inputs)   
    x=layers.Conv2D(16,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)  
    print('none1',x.shape)
    x = keras.layers.BatchNormalization()(x)
    x1= keras.layers.ReLU()(x)
    print('none2',x.shape)
    x=deepwise2d(x1,k=6,ratio=channelratio,cbrm=False,name='zero') 
    x=layers.concatenate([inputs,x],axis=-1) 
  
    x2=layers.Conv2D(32,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)  
    # x2=layers.MaxPooling2D((2,2))(x)
    x=deepwise2d(x2,k=6,ratio=channelratio,cbrm=False,name='one')
    x=layers.concatenate([inputs,x],axis=-1) 
    x3=layers.Conv2D(64,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)  
    # x3=layers.MaxPooling2D((2,2))(x)
    x4=deepwise2d(x3,k=6,ratio=channelratio,cbrm=True,name='two')
    x4=layers.concatenate([inputs,x4],axis=-1) 
    x=layers.Conv2D(96,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x4) 
    # x4=layers.MaxPooling2D((2,2))(x)
   
    if sar==False:
        x=deepwise2d(x,k=6,ratio=channelratio,cbrm=True,name='three')
        x=layers.concatenate([inputs,x],axis=-1) 
        x=layers.Conv2D(128,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
        x5=deepwise2d(x,k=6,ratio=channelratio,cbrm=True,name='four')
        x=layers.concatenate([inputs,x5],axis=-1) 
    # x=layers.Conv2D(256,(3,3),strides=(2,2),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    # x=deepwise2d(x,filter1=128,k=2)
    
    # x1=ResBlock(x,channel=64,geoinputs=geoinputs,two=True,geodateattention=True,attentionbefore=True,attentionafter=False)
    # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxx',x.shape)
    # x1=layers.MaxPooling2D((2,2))(x1)
    # x2=ResBlock(x1,channel=128,geoinputs=geoinputs,two=True,geodateattention=True,attentionbefore=True,attentionafter=False)
    # # # 参数量很大
    # x=layers.concatenate([x1,x2],axis=-1) 
    # x=layers.MaxPooling2D((2,2))(x)
    # x3=ResBlock(x,channel=128,geoinputs=geoinputs,two=True,geodateattention=True,attentionbefore=False,attentionafter=True)
    # x=layers.concatenate([x,x3],axis=-1) 
    # x=layers.MaxPooling2D((2,2))(x)
    # print('xxxxxxxxxxxxxxxxxxxxxxxxxxx',x.shape) #(None, 1, 1, 320)
    if cnn1d==True:
        # x=Reshape((x.shape[1],x.shape[3],x.shape[2]))(x)
        
        x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    
    # x=eca_block(x, b=1, gama=2)
    # x=GlobalAveragePooling2D()(x)
    # x=Flatten()(x)
    
    # 在高度和宽度轴上重复张量，变为 (None, 11, 11, 1)
   
    # dense1=Dense(dense1.shape[-1],activation='relu',kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(dense1)
    # dense2=Dense(dense2.shape[-1],activation='relu',kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(dense2)
    # dense3=Dense(dense3.shape[-1],activation='relu',kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(dense3)
    # dense4=Dense(dense4.shape[-1],activation='relu',kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(dense4)
    # dense=layers.concatenate([dense1,dense2,dense3,dense4],axis=-1) 
    # dense = tf.tile(dense, [1, 11, 11, 1])
    # x=layers.concatenate([x,dense],axis=-1)
    # print(dense.shape)
    # x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    return x
# inputshape=(11,11,9)
# geotime=(4,)
# inputs=keras.Input(shape=inputshape)
# geoinputs=keras.Input(shape=geotime)
# m=singlebranch(inputs,geoinputs,channelratio=8,cnn1d=False)
# model=Model(inputs=[inputs,geoinputs],outputs=m)
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
# inputshape=(2,11,11,17)
# numFilters=[16,32,64,128,256,512]
# geoinput=(3,)
# model=DualCnn2dGeotimeCbrm(inputshape,channelratio=4,dual=True,bandswithindex=True,dropout=0,L2=0.002) # ratio越小参数越大
# model.compile(optimizer=optimizer ,loss= cross_loss, metrics=metrics)
# print(model.summary())

def medianCnn2d(inputshape,channelratio,bandswithindex=False,bandposition=14,indexposition=17,sar=False,dropout=0.2,L2=0):
    '''
    inputshape: (2,patchsize,patchsize,15). last three channels are doy,lat,lon
    channelratio:the ratio of channel atttention dense layer after max and average pooling
    '''
    if bandswithindex:
        bands=indexposition
    else:
        bands=bandposition

    if L2>0:
        reg = tf.keras.regularizers.l2(l=L2)
    else: 
        reg = tf.keras.regularizers.l2(l=0.0)
    inputs= keras.Input(shape=inputshape) #(None, 2, 11, 11, 15)
    print('inputs',  inputs.shape)
    s2=layers.Lambda(lambda x: x[...,5:bands])(inputs)
    s2=singlebranch(s2,channelratio,cnn1d=True,sar=False)
    if sar==True:
        s1=layers.Lambda(lambda x: x[...,2:4])(inputs)
        vhmin= layers.Lambda(lambda x: x[...,2:3])(inputs)
        vhmax= layers.Lambda(lambda x: x[...,3:4])(inputs)
        ratio = layers.Lambda(lambda x: x[1] / x[0])([vhmin, vhmax])
        s1=layers.concatenate([s1,ratio],axis=-1)
        s1=singlebranch(s1,channelratio,cnn1d=True,sar=False)
        x=s1#layers.concatenate([s1,s2],axis=-1)
    else:
        x=s2
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
    x=channel_attention(x, 8,False,False) #inputs,geoinputs,ratio=8,geo=True,dense=False
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
#%%
# inputshape=(11,11,14)
# channelratio=8
# model=medianCnn2d(inputshape,channelratio,bandswithindex=False,sar=True,dropout=0.2,L2=0)
# model.summary()
# def cnn1d(x):
#     x= keras.Input(shape=x)
#     kernel_initializer = tf.keras.initializers.Ones()
#     # y= layers.Conv1D(1,
#     #                         3,
#     #                         padding='same',
#     #                         dilation_rate=1,
#     #                         strides=1, kernel_initializer=kernel_initializer
#     #                         )(x)
#     y = keras.layers.Conv1D(32, (1,),padding='valid', activation='relu')(x)
#     return Model(x,y)
# x3=(1,1,15)
# x=(10,1)
# x1=(10,5)
# # model=cnn1d(x)#.summary()
# # model1=cnn1d(x1)#.summary()
# model3=cnn1d(x3)#.summary()
# print(model3.summary())
# # 打印每一层的权重
# for layer in model3.layers:
#     print(f"\nLayer: {layer.name}")
#     weights = layer.get_weights()
#     if weights:
#         for w in weights:
#             print(w.shape)
#             print(w)
#     else:
#         print("No weights for this layer")
# x = tf.random.normal((1,10,5))
# print(x)
# output = model3.predict(x)
# print("\nModel output:")
# print(output.shape) #(1, 8, 32)
# x = tf.constant([1, 2, 3, 4, 5], dtype=np.float32)
# x = tf.reshape(x, (1,1, 5))
# kernel_initializer = tf.keras.initializers.Ones()

# conv_with_stride = layers.Conv1D(2,
#                         3,
#                         padding='valid',
#                         dilation_rate=1,
#                         strides=2, 
#                         kernel_initializer=kernel_initializer)
# y = conv_with_stride(x)
# print(y)



# model=DualCnn2dGeotimeCbrm(inputshape,4,0,0.002,True) # ratio越小参数越大
# print(model)
# print(model.summary())
# shape_x = (2, 2,11, 11, 16)
# shape_y = (2, 1)
# # Create random data for x (assuming you want random values)
# x = np.random.rand(*shape_x)
# # Create random integers for y (0 or 1)
# y = np.random.randint(2, size=shape_y)
# y= tf.convert_to_tensor(y, dtype=tf.float32)
# y=to_categorical(y, num_classes=2)
# model.compile(optimizer=optimizer ,loss= cross_loss, metrics=metrics)
# history=model.fit(x, y,epochs=10,
# verbose=2,callbacks=callback_,shuffle=False)

# print(model.summary())
# # model=tf.keras.models.load_model(saveModelPath,custom_objects={"K": K})
# start=time.time()
# # callback_= [
# #     keras.callbacks.ModelCheckpoint(
# #         monitor='val_loss',
# #         filepath=saveModelPath,
# #         mode='min',
# #         save_best_only=True,
# #         save_weights_only=False,
# #         verbose=2
# #     )]
# # Splitting the data into training (70%), validation (20%), and test (10%) sets
# xtrain, X_temp, ytrain, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)
# xval, xtest, yval, ytest = train_test_split(X_temp, y_temp, test_size=1/2, random_state=42)
# print(xtrain.shape,xval.shape,yval.shape)
# print(callback_)
# # history=model.fit(xtrain, ytrain,epochs=30,validation_data=(xval, yval),
# # batch_size= batch_size,verbose=2,shuffle=True)
# # validation_data=None,可以调用callbacks
# # validation_data=(xval, yval),不调用callbacks，可以fit
# # callback可以保存csv 精度变化，但不能保存模型和训练和损失曲线



# %%
