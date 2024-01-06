
from semantic_segmentation.model.module import *
# from semantic_segmentation.dataset.datagenerate import get_c_value
# print(get_c_value())
def singlebranch(inputs,geoinputs,channelratio,cnn1d=False):
   
    x=layers.Conv2D(32,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(inputs)   
    x=layers.Conv2D(32,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(inputs)  
    x = keras.layers.BatchNormalization()(x)
    x= keras.layers.ReLU()(x)
    x1=ResBlock(x,channel=64,geoinputs=geoinputs,two=True,geodateattention=True,attentionbefore=False,attentionafter=True)
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxx',x.shape)
    x1=layers.MaxPooling2D((2,2))(x1)
    x2=ResBlock(x1,channel=128,geoinputs=geoinputs,two=True,geodateattention=True,attentionbefore=False,attentionafter=True)
    # # 参数量很大
    x=layers.concatenate([x1,x2],axis=-1) 
    x=layers.MaxPooling2D((2,2))(x)
    x3=ResBlock(x,channel=256,geoinputs=geoinputs,two=False,geodateattention=True,attentionbefore=False,attentionafter=True)
    x=layers.concatenate([x,x3],axis=-1) 
    x=layers.MaxPooling2D((2,2))(x)
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxx',x.shape)
    if cnn1d==True:
        x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    # x=layers.concatenate([inputs,x],axis=-1) 
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
    inputtime1geotimeinputs= layers.Lambda(lambda x: x[...,0:1,5:6,5:6,12:16])(inputs) # (1,1,3)
    inputtime2geotimeinputs= layers.Lambda(lambda x: x[...,1:2,5:6,5:6,12:16])(inputs) # (1,1,3)
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
    time1cnn=singlebranch(inputtime1bands,inputtime1geotimeinputs,channelratio,cnn1d=True)

    time2cnn=singlebranch(inputtime2bands,inputtime2geotimeinputs,channelratio,cnn1d=True)
    x=layers.concatenate([time1cnn,time2cnn],axis=-1) #add()?
    x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    x=layers.Conv2D(128,(3,3),strides=(2,2),padding='same',dilation_rate=(1, 1),kernel_initializer='he_normal', use_bias=False)(x) 
    # x = layers.MaxPooling2D(strides=2, padding="same")(x)
    # x = keras.layers.BatchNormalization()(x)
    # x= keras.layers.ReLU()(x)
    # x=channel_attention(x,,channelratio,False)
    # x=spatial_attention(x)
    # x=layers.Conv2D(256,(3,3),strides=(1,1),padding='same',dilation_rate=(2, 2),kernel_initializer='he_normal', use_bias=False)(x) 
    x = keras.layers.BatchNormalization()(x)
    # x = layers.MaxPooling2D(strides=2, padding="same")(x)
    x= keras.layers.ReLU()(x)
    # x=channel_attention(x,x,channelratio,False)
    # x=spatial_attention(x)
    # x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    # x=layers.AveragePooling2D(pool_size=3)(x)
    # x=layers.GlobalAveragePooling2D()(x)
    x=Flatten()(x)
    # x=Dense(512,kernel_regularizer=reg)(x)
    x=Dense(256,kernel_regularizer=reg)(x)
    x= layers.Dropout(dropout)(x)
    print('x',x.shape)
    output_layer=Dense(2,kernel_regularizer=reg,activation='softmax')(x) #,activation='sigmoid'
    return Model(inputs,output_layer)
#%%
from sklearn.model_selection import train_test_split
from cfgs import *
inputshape=(2,11,11,16)
numFilters=[16,32,64,128,256,512]
geoinput=(3,)
model=DualCnn2dGeotimeCbrm(inputshape,channelratio=4,dual=True,dropout=0,L2=0.002) # ratio越小参数越大
model.compile(optimizer=optimizer ,loss= cross_loss, metrics=metrics)
print(model.summary())
# model=DualCnn2dGeotimeCbrm(inputshape,4,0,0.002,True) # ratio越小参数越大
# print(model)
# print(model.summary())
# shape_x = (500, 2,11, 11, 16)
# shape_y = (500, 1)

# # Create random data for x (assuming you want random values)
# x = np.random.rand(*shape_x)

# # Create random integers for y (0 or 1)
# y = np.random.randint(2, size=shape_y)
# y= tf.convert_to_tensor(y, dtype=tf.float32)
# y=to_categorical(y, num_classes=2)
# model.compile(optimizer=optimizer ,loss= cross_loss, metrics=metrics)
# # print(model.summary())
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

# history=model.fit(xtrain, ytrain,epochs=10,validation_data=(xval, yval),
# batch_size= 16,verbose=2,callbacks=callback_,shuffle=False)
