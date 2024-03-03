#%%
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
from RF_DL.model.loss import *
from RF_DL.plot import *
import logging
import inspect
# def reset_random_seeds():
#    os.environ['PYTHONHASHSEED']=str(999)
#    random.seed(999)
#    np.random.seed(999)
#    tf.random.set_seed(999)
# reset_random_seeds()
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


def channel_attention_lowhigh(inputslow,inputshigh,ratio=8):
    
    # 通道维度上的平均池化
    # avg_pool= layers.TimeDistributed(layers.GlobalAveragePooling2D())(input_feature)
    inputs=layers.concatenate([inputslow,inputshigh],axis=-1)
    avg_pool= layers.GlobalAveragePooling2D()(inputs)
    
    max_pool = layers.GlobalMaxPooling2D()(inputs)


   
    avg_pool=Reshape((1,1,avg_pool.shape[-1]))(avg_pool)

    # avgcatcenter=Concatenate(axis=-1)([avg_pool,inputs])
    max_pool = Reshape((1,1,max_pool.shape[-1]))(max_pool)
    # maxcatcenter=Concatenate(axis=-1)([max_pool,inputs])

    # avg_pool = Lambda(lambda x: K.reshape(x, (K.shape(x)[0], 1, 1, K.shape(x)[-1])))(avggeo_pool)
    # max_pool= Lambda(lambda x: K.reshape(x, (K.shape(x)[0], 1, 1, K.shape(x)[-1])))(maxgeo_pool)
    # print('avg_pool',avg_pool.shape)
    channel =  avg_pool.shape[-1]  # 获取通道维度
    # print('channel',channel)
    # print('rtatio',ratio)
    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    avg_pool=shared_layer_one(avg_pool
                               )
    avg_pool=shared_layer_two(avg_pool)
    print('avg2', avg_pool.shape) 
    # print( max_pool.shape)
 
    max_pool= shared_layer_one(max_pool)
    max_pool= shared_layer_two(max_pool)
    # print('ma', max_pool2.shape) 
    # 通道注意力的输出
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    inputshigh=Conv2D(filters=cbam_feature.shape[-1], kernel_size=1, strides=1, padding='same',  kernel_initializer='he_normal', use_bias=False)(inputshigh)
    inputshigh=BatchNormalization()(inputshigh)
    # cbam_feature=Conv2D(filters=inputshigh.shape[-1], kernel_size=1, strides=1, padding='same',  kernel_initializer='he_normal', use_bias=False)(cbam_feature)
    return multiply([inputshigh, cbam_feature])
def spatial_attention_lowhigh(low,high):
    # 空间注意力的计算，这里可以根据需要修改
    kernel_size = 5
    #两者都没有在网络图中显示
    # inputs=layers.concatenate([low,high],axis=-1)
    input1s=Concatenate(axis=-1)([low,high])
    print('inputconctat',input1s.shape)
    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input1s)
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input1s)
    print('concatspatial',avg_pool.shape,max_pool.shape) #(None, 256, 256, 1) (None, 256, 256, 1)
    concat = Concatenate(axis=-1)([avg_pool, max_pool]) #(None, 256, 256, 1) (None, 256, 256, 2)

    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    # print("spation",cbam_feature.shape) #(None, 7,7 1)
    # low=Conv2D(filters=cbam_feature.shape[-1], kernel_size=1, strides=1, padding='same',  kernel_initializer='he_normal', use_bias=False)(low)
    # low=BatchNormalization()(low)
    return multiply([low, cbam_feature])
def deepwise2d(inputs,k,ratio=8,sequentialcbrm=True,cbrmlowhigh=False,coord=False,name='one'):
    x1=layers.Conv2D(inputs.shape[-1]*k,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(inputs) 
    x1=layers.BatchNormalization()(x1)
    x1= layers.ReLU(max_value=6)(x1)
    x=DepthwiseConv2D(3,padding='same',activation='relu')(x1)
    # x=DepthwiseConv2D(3,padding='same',activation='relu')(x)
    # x=coordinate(x,ratio=8)
    x=layers.Conv2D(inputs.shape[-1],(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    # x=Add()([inputs,x])
    print('dpwise',x.shape)
    if sequentialcbrm:
        if cbrmlowhigh:
            x=channel_attention_lowhigh(inputs,x,ratio=ratio)
            x=spatial_attention_lowhigh(inputs,x)
        else:
            x=channel_attention(x,ratio=ratio)
            x=spatial_attention(x)
    if coord:
        x=coordinate(x,ratio=32)
    # x=layers.concatenate([inputs,x],axis=-1) 
    x=layers.add([inputs,x])
    return x

def depthwiseattention(inputs,channelratio=8,depthwisecbrm=False,cbrmlowhigh=False,depthwisecorrd=False):
    print('input',inputs.shape)
  
    x=layers.Conv2D(16,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(inputs)   
    x=layers.Conv2D(16,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)  
    print('none1',x.shape)
    x = keras.layers.BatchNormalization()(x)
    x1= keras.layers.ReLU()(x)
    print('none2',x.shape)
    x=deepwise2d(x1,k=6,ratio=channelratio,sequentialcbrm=False,cbrmlowhigh=cbrmlowhigh,coord=False,name='zero') 
    # x=layers.concatenate([inputs,x],axis=-1) 
    x2=layers.Conv2D(32,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)  
    # x2=layers.MaxPooling2D((2,2))(x)
    x=deepwise2d(x2,k=6,ratio=channelratio,sequentialcbrm=False,cbrmlowhigh=cbrmlowhigh,coord=False,name='one')
    # x=layers.concatenate([inputs,x],axis=-1) 
    x3=layers.Conv2D(64,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)  
    # x3=layers.MaxPooling2D((2,2))(x)
    x=deepwise2d(x3,k=6,ratio=channelratio,sequentialcbrm=False,cbrmlowhigh=cbrmlowhigh,coord=depthwisecorrd,name='two')
    # x=layers.concatenate([inputs,x],axis=-1) 
    x=layers.Conv2D(96,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    # x4=layers.MaxPooling2D((2,2))(x)
    x=deepwise2d(x,k=6,ratio=channelratio,sequentialcbrm=False,cbrmlowhigh=cbrmlowhigh,coord=depthwisecorrd,name='three')
    # x=layers.concatenate([inputs,x],axis=-1) 
    x=layers.Conv2D(128,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    x=deepwise2d(x,k=6,ratio=channelratio,sequentialcbrm=False,cbrmlowhigh=cbrmlowhigh,coord=depthwisecorrd,name='four')
    # x=deepwise2d(x,k=6,ratio=channelratio,sequentialcbrm=depthwisecbrm,cbrmlowhigh=cbrmlowhigh,coord=depthwisecorrd,name='four')
    # x=layers.concatenate([inputs,x5],axis=-1) 
    # if cnn1d==True:
    #     # x=Reshape((x.shape[1],x.shape[3],x.shape[2]))(x)
        
    #     x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    # x=eca_block(x, b=1, gama=2)
    # x=GlobalAveragePooling2D()(x)
    # x=Flatten()(x)
    # x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    return x

def medianCnn2d(inputshape,channelratio, inputtag,depthwisecbrm,cbrmlowhigh,concatlowhigh,dropout,L2):
    if inputtag==0:
        startbands,endbands=0,inputshape[-1]
    if  inputtag==1:
        startbands,endbands=0,13
    if  inputtag==2:
        startbands,endbands=13,26
    if  inputtag==3:
         startbands,endbands=26,39
    if L2>0:
        reg = tf.keras.regularizers.l2(l=L2)
    else: 
        reg = tf.keras.regularizers.l2(l=0.0)
    inputs= keras.Input(shape=inputshape) #(None, 2, 11, 11, 15)
    
    x1=layers.Lambda(lambda x: x[...,startbands:endbands])(inputs)
    inputheight,inputwidth,inputschannels=x1.shape[1],x1.shape[2],x1.shape[-1]
    x=depthwiseattention(x1,channelratio=channelratio,depthwisecbrm=depthwisecbrm,cbrmlowhigh=cbrmlowhigh,depthwisecorrd=depthwisecorrd)
    x=layers.Conv2D(256,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
  
    # if concatlowhigh==True:
    #     input1s=Conv1D(256,(1,),strides=(1,),padding='same',kernel_initializer='he_normal', use_bias=False)(inputs)
    #     input1s= keras.layers.BatchNormalization()(input1s)
    #     inp=layers.concatenate([x,input1s],axis=-1)
    #     x=channel_attention(inp,ratio=8)
    #     x=spatial_attention(x)
    #     x=Add()([input1s,x])
    # if concat==True:
    #     xcenter=layers.Lambda(lambda x: x[...,inputheight//2:inputheight//2+1,inputwidth//2:inputwidth//2+1,0:inputschannels])(inputs)
    #     # xcenteravg=layers.GlobalAveragePooling2D()(inputs)
    #     # xcentermax=layers.GlobalMaxPooling2D()(inputs)
    #     # xcenter=layers.add([xcenteravg,xcentermax])
    #     xcenter=Reshape((xcenter.shape[-1],))(xcenter)
    #     xcenter=Dense(256,kernel_regularizer=reg)(xcenter)
    # else:
    #     inp=x
    #     x=channel_attention(inp,ratio=8)
    #     x=spatial_attention(x)
    x=layers.Conv2D(256,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # x=layers.Conv2D(256,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = layers.ReLU()(x)
    x = layers.MaxPooling2D(strides=2, padding="same")(x)
    x=layers.GlobalAveragePooling2D()(x)
    x=Flatten()(x)
    x=Dense(256,kernel_regularizer=reg)(x)
    if concat==True:
        xcenter=layers.Lambda(lambda x: x[...,inputheight//2:inputheight//2+1,inputwidth//2:inputwidth//2+1,0:inputschannels])(inputs)
        xcenter=Reshape((xcenter.shape[-1],))(xcenter)
        xcenter=Dense(256,kernel_regularizer=reg)(xcenter)
        x=layers.concatenate([xcenter,x],axis=-1)
    x= layers.Dropout(dropout)(x)
    x=Dense(64,kernel_regularizer=reg)(x)
    x= layers.Dropout(dropout)(x)
    print('x',x.shape)
    output_layer=Dense(2,kernel_regularizer=reg,activation='softmax')(x) #,activation='sigmoid'
        # output_layer=Dense(1, activation='sigmoid')(x)

    return Model(inputs,output_layer)

#%%    
def dataprogress(savetrainxPath, savetrainyPath, savevalxPath, savevalyPath,median12max3,median12max4,median13max3,median13max4):
    xtrain,xval=np.load(savetrainxPath),np.load(savevalxPath)
    patchshape=xtrain.shape[1]
    
    if median13max3:
        xmedian=xtrain[...,39:52]
        xmax=xtrain[...,-3:]
        xtrain=np.concatenate([xmedian,xmax],axis=-1)
        xvalmedian=xval[...,39:52]
        xvalmax=xval[...,-3:]
        xval=np.concatenate([xvalmedian,xvalmax],axis=-1)
    elif median13max4:
        xmedian=xtrain[...,39:52]
        xmax=xtrain[...,-3:]
        xb11max=xtrain[...,-5:-4]
        xtrain=np.concatenate([xmedian,xb11max,xmax],axis=-1)
        xvalmedian=xval[...,39:52]
        xvalb11max=xval[...,-5:-4]
        xvalmax=xval[...,-3:]
        xval=np.concatenate([xvalmedian,xvalb11max,xvalmax],axis=-1)
    elif median12max3:
        xmedian=xtrain[...,40:52]
        xmax=xtrain[...,-3:]
        xtrain=np.concatenate([xmedian,xmax],axis=-1)
        xvalmedian=xval[...,40:52]
        xvalmax=xval[...,-3:]
        xval=np.concatenate([xvalmedian,xvalmax],axis=-1)
    elif median12max4:
        xmedian=xtrain[...,40:52]
        xmax=xtrain[...,-3:]
        xb11max=xtrain[...,-5:-4]
        xtrain=np.concatenate([xmedian,xb11max,xmax],axis=-1)
        xvalmedian=xval[...,40:52]
        xvalb11max=xval[...,-5:-4]
        xvalmax=xval[...,-3:]
        xval=np.concatenate([xvalmedian,xvalb11max,xvalmax],axis=-1)
    channels=xtrain.shape[-1]
    xtrain=xtrain[...,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,0:channels]
    xval=xval[...,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,0:channels]
    
   
    # xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=6/8, random_state=42)
    ytrain=np.load(savetrainyPath) #loadandprocessdata(savetrainxPath,savetrainyPath)
    yval= np.load(savevalyPath) #loadandprocessdata(savevalxPath,savevalyPath)
    # xtrain, X_temp, ytrain, y_temp = train_test_split(xtrain, ytrain, test_size=0.85, random_state=42)
    ytrain=np.where(ytrain!=0,1,ytrain)
    ytrain= tf.convert_to_tensor(ytrain, dtype=tf.float32)
    ytrain=to_categorical(ytrain, num_classes=2)
    yval=np.where(yval!=0,1,yval)
    yval= tf.convert_to_tensor(yval, dtype=tf.float32)
    yval=to_categorical(yval, num_classes=2)
    
    x=np.concatenate([xtrain,xval],axis=0)
    y=np.concatenate([ytrain,yval],axis=0)
    print(x.shape,y.shape)
    mean=np.nanmean(x,axis=(0,1,2))
    std=np.nanstd(x,axis=(0,1,2))
   
    xtrain=np.where(np.isnan(xtrain), 0, (xtrain - mean) / std)
    xval=np.where(np.isnan(xval), 0, (xval- mean) / std)
    meanpath=os.path.join( savemeanstddir,f'sample{xtrain.shape[0]}x{xval.shape[0]}_mean{mean.shape[0]}.npy')#os.path.dirname(savetrainxPath)
    stdpath=os.path.join(savemeanstddir,f'sample{xtrain.shape[0]}x{xval.shape[0]}_std{std.shape[0]}.npy')
    np.save(meanpath,mean)
    np.save(stdpath,std)
    return xtrain,xval,ytrain,yval,meanpath,stdpath,xtrain.shape[0],xval.shape[0]
   
def train( modeltag,depthwisecbrm,cbrmlowhigh,depthwisecorrd,ecabeforedense,cbrmbeforedense,concat,dropout,l2,agu,epochs,weight=False,    learning_rate=0.0001
    ):
        start=time.time()
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        if modeltag=='depthwiseadd_nocat':
            model=medianCnn2d(inputshape=(h,w,c),channelratio=8,inputtag=inputtype,depthwisecbrm=depthwisecbrm,cbrmlowhigh=cbrmlowhigh,concatlowhigh=concat,dropout=dropout,L2=l2)#inputshape,channelratio,bandswithindex=False,bandposition=14,indexposition=17,sar=False,dropout=0.2,L2=0simplecnn2d(inputshape=(h,w,c),num_filters=[32,64,128,256,256],dropratio=0)
            print(model.summary())
            if weight:
                loss=binary_weighted_cross_entropy
            else:
                loss=cross_loss
            
            model.compile(optimizer=optimizer ,loss=loss, metrics=['accuracy'])
            start=time.time()
            from datetime import datetime

            # current_date = datetime.now().strftime('%Y-%m-%d')
         
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
            # current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            current_date =      current_date .replace(' ', '-').replace(':', '-')
            print(     current_date )

            print(current_date)
            # log
            saveVersion=f'{inputtag}(13)train{traincounts}val{valcounts}_agu{agu}weght{weight}_{modeltag}cbrmn{depthwisecbrm}_cbrmlowhigh{cbrmlowhigh}__catlowhigh{concat}_drop{dropout}_l2{l2}' #centerpixelcatcenter
        if modeltag=='simplecnn2d':
            numfilters=[32,64,128,256,256]
            model=simplecnn2d(inputshape=(h,w,c),num_filters=numfilters,dropratio=0)
            if weight:
                loss=binary_weighted_cross_entropy
            model.compile(optimizer=optimizer ,loss= loss, metrics=['accuracy'])
            layername='x'.join(map(str, numfilters))
            cbrm,coord,eca=0,0,0
            saveVersion=f'{modeltag}_{layername}_cbrm{cbrm}_coord{coord}_eca{eca}'
        
        workEnv= 'D:/ricemodify/runRFDL/train'
        checkpointDir = os.path.join(workEnv, f'{year}{inputtag}patchsize{patchsize}/log/{saveVersion}') #/train/model is wrong
        plotdir=os.path.join(workEnv, f'{year}{inputtag}patchsize{patchsize}/plot')
        resumeModelDir= os.path.join(checkpointDir, 'resume')
        os.makedirs(checkpointDir, exist_ok=True) # os.makedirs(logsDir, exist_ok=True)
        os.makedirs(resumeModelDir, exist_ok=True)
        os.makedirs(plotdir,exist_ok=True)
        plotpath=os.path.join(plotdir,f'{current_date}_{saveVersion}.jpg')
        plotpath2=os.path.join(checkpointDir,f'{current_date}_{saveVersion}.jpg')
        resumeModelpath=os.path.join(resumeModelDir, f'{current_date}_{saveVersion}.h5')
        saveModelPath = os.path.join(checkpointDir,  f'{current_date}_{saveVersion}.h5')
        print('saveModelPath',saveModelPath)
        # plotPath = os.path.join(checkpointDir, f'{savemeanstdtag}_lossaccuracy.png')
        confusion_matrixpath=os.path.join(checkpointDir, f'{saveVersion}_confusion_matrix.png')
        callback_ = [
                    callbacks.ModelCheckpoint(monitor='val_accuracy', filepath=saveModelPath, mode='max',
                                            save_best_only=True, save_weights_only=False, verbose=1),
                    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=plateauPatience,
                                                min_delta=5e-3, min_lr=1e-6, verbose=1),
                    callbacks.CSVLogger(filename=os.path.join( checkpointDir ,  current_date + '_' +saveVersion+ '.csv')),
                    # TensorBoard(log_dir=logsDir_,histogram_freq=1)
            ]
  
        if agu:
            mytrain=dataagument(xtrain,ytrain,64)
            steps_per_epoch=len(xtrain)/64
           
            history = model.fit_generator(mytrain, steps_per_epoch=steps_per_epoch,validation_data=(xval,yval),
                    epochs=epochs,callbacks=callback_,verbose=2,shuffle=True)

               
            # if weight:
            #     history = model.fit(xtrain,ytrain, validation_data=(xval,yval),batch_size=64,class_weight=compute_class_weights(ytrain),
            #         epochs=epochs,callbacks=callback_,verbose=2,shuffle=True)
            #     model=  tf.keras.models.load_model(saveModelPath,custom_objects={"K": K,'binary_weighted_cross_entropy':binary_weighted_cross_entropy})
        else:
                history = model.fit(xtrain,ytrain, validation_data=(xval,yval),batch_size=64,
                    epochs=epochs,callbacks=callback_,verbose=2,shuffle=True)

        end=time.time()
        total=(end-start)/60
        # Configure logging
        log_file_path=os.path.join(checkpointDir, saveVersion+'.txt')
        logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()  # To also print logs to console
        ])
        total=(time.time()-start)/60
        # logging.getLogger().setLevel(logging.INFO)
        logging.info(f'训练{epochs}轮花费的时间: {total}分')
        logging.info('trainsamplecount: %s',  traincounts)
        logging.info('valsamplecount: %s',  valcounts)
        logging.info('savetrainxPath: %s',  savetrainxPath)
        logging.info('savetrainyPath: %s',  savetrainyPath)
        logging.info(' savevalxPath: %s',  savevalxPath)
        logging.info('savevalyPath: %s',  savevalyPath)
        logging.info('saveMeanPath:%s', meanpath)
        logging.info('saveStdPath:%s', stdpath)
        logging.info('data_augmentation: %s',agu)
        logging.info('img_height: %s',h)
        logging.info('img_width: %s', w)
        logging.info('img_channels: %s', c)
        logging.info('dropout:%s', dropout) # if use logging.info('dropout:\%s', dropout),the result is dropout:\0
        logging.info('l2:%s', l2)
        logging.info('learning_rate: %s', learning_rate)
        logging.info('batch_size: %s', batch_size)
        weight=True
        if modeltag=='simplecnn2d':
            model_function_source = inspect.getsource(simplecnn2d)
            print(model_function_source)
            logging.info('Model Function Source Code:\n%s',  model_function_source)
        if modeltag=='depthwiseadd_nocat':
            model_function_source = inspect.getsource(medianCnn2d)
            print(model_function_source)
            logging.info('Model Function Source Code:\n%s',  model_function_source)
        plot(history,saveVersion,plotpath,plotpath2)
        return 0

if __name__ == "__main__":
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()

    # savetrainxPath=r"D:\ricemodify\datasetRF\2022medianq75maxpatchsize11\trainxgrid61324nomeanstd_10708x11x11x39.npy"#"D:\ricemodify\datasetRF\pathsize11\traingrid61324\new2022trainxgrid61324nomeanstd_10708x11x11x9.npy"
    # savetrainyPath=  r"D:\ricemodify\datasetRF\2022medianq75maxpatchsize11\trainygrid61324_10708x11x11x5.npy"#"D:\ricemodify\datasetRF\pathsize11\traingrid61324\new2022trainygrid61324_10708x11x11x2.npy"
    # savevalxPath= r"D:\ricemodify\datasetRF\2022medianq75maxpatchsize11\valxgrid1729nomeanstd_7031x11x11x39.npy"#"D:\ricemodify\datasetRF\pathsize11\valgrid1729\new2023valxgrid1729nomeanstd_7031x11x11x9.npy"
    # savevalyPath= r"D:\ricemodify\datasetRF\2022medianq75maxpatchsize11\valygrid1729_7031x11x11x5.npy"#"D:\ricemodify\datasetRF\pathsize11\valgrid1729\new2023valygrid1729_7031x11x11x2.npy"
    
    # savetrainxPath=r"D:\ricemodify\datasetRF\2022sevenperpatchsize15\trainxgrid61324nomeanstd_10691x15x15x91.npy"
    # savetrainyPath=r"D:\ricemodify\datasetRF\2022sevenperpatchsize15\trainygrid61324_10691x15x15x5.npy"
    # savevalxPath=r"D:\ricemodify\datasetRF\2022sevenperpatchsize15\valxgrid1729nomeanstd_6995x15x15x91.npy"
    # savevalyPath=r"D:\ricemodify\datasetRF\2022sevenperpatchsize15\valygrid1729_6995x15x15x5.npy"
    savetrainxPath=r"D:\ricemodify\datasetRF\2022patchsize15\sevenper\trainxgrid61324nomeanstd_6265x15x15x91.npy"
    savetrainyPath=r"D:\ricemodify\datasetRF\2022patchsize15\sevenper\trainygrid61324_6265x15x15x5.npy"
    savevalxPath=r"D:\ricemodify\datasetRF\2022patchsize15\sevenper\valxgrid1729nomeanstd_4103x15x15x91.npy"
    savevalyPath=r"D:\ricemodify\datasetRF\2022patchsize15\sevenper\valygrid1729_4103x15x15x5.npy"
    # 数据选择
    # 0:meadianq75max 1:median 2:Q75 3:max
    patchsize=11
    year=2022
    inputtag='12median3max'#'sevenper'#'medianq75max'
    # savemeanstdtag=f'{year}_{inputtag}(10bands3index)'
    savemeanstddir=rf"D:\ricemodify\datasetRF\{year}{inputtag}patchsize{patchsize}"
    os.makedirs(savemeanstddir,exist_ok=True)
    median12max3,median12max4,median13max3,median13max4=True,False,False,False
    if inputtag=='medianq75max' or inputtag=='sevenper'or inputtag=='13median3max' or inputtag=='13median4max' or inputtag=='12median3max' or inputtag=='12median4max':
        inputtype=0
     
    if inputtag=='median':
          inputtype=1
     
    if inputtag=='q75':
          inputtype=2
     
    if inputtag=='max':
          inputtype=3
  
    xtrain,xval,ytrain,yval,meanpath,stdpath,traincounts,valcounts= dataprogress(savetrainxPath, savetrainyPath, savevalxPath, savevalyPath,median12max3,median12max4,median13max3,median13max4)
    # 超参数
    batch_size=64
  

    plateauPatience=5
    dropout,l2=0,0

    # 定义指数衰减函数
    # def lr_schedule(epoch, lr):
    #     decay_factor = 0.95  # 每个epoch学习率衰减的因子
    #     if epoch % 10 == 0 and epoch > 0:
    #         return lr * decay_factor
    #     return lr

    cross_loss=keras.losses.CategoricalCrossentropy(from_logits=False)
    cosineloss=tf.keras.losses.CosineSimilarity(axis=1)
    # 模型结构
    depthwisecbrm=True
    ecabeforedense=False
    cbrmbefordense=False
    depthwisecorrd=False
    concat=False
    weight=True
    # 损失函数
    loss=cross_loss#binary_weighted_cross_entropy
    h,w,c=xtrain.shape[1],xtrain.shape[2],xtrain.shape[-1]
    start=time.time()
    
      
        # logging.info('Model Structure:')
        # model.summary(print_fn=lambda x: logging.info(x))
        # print(model.summary())
        # model.compile(optimizer=optimizer ,loss= 'mse', metrics='mae')
        # loss='mse', metrics=['mae']
        # compose_transform = Compose([RandomRotation(),RandomVerticalFlip(),RandomHorizontalFlip()])#RandomContrast(),transformer.RandomBrightness()
        # Log training results
  

        
    depthwisecbrm=None
    epochs=20
    ecabeforedense=0
    cbrmbeforedense=1
    depthwisecorrd=0
    concat=None
    agu=1
    weight=True
    modeltag='depthwiseadd_nocat'#'simplecnn2d' #
    # train(modeltag,depthwisecbrm,depthwisecorrd,ecabeforedense,cbrmbeforedense,concat,dropout,l2,agu)
    # 2024-1-31-11.15 测试训练集0.3  weight=True agu=Fasle depthwisecbrm low high features cat-cnn1d and cnn2d

    # depthwisecbrm3=true,cbrmbeforedense=0,cat=False效果很差，初始训练精度0.6，cat为true,初始训练精度0.8，但是还不如depthwisecbrm3False,cat=False的0.86
    # depthwisecbrm3 ,cbrmbefordense=0,0,cat=0好于depthwisecbrm ,cbrmbefordense=0,0,cat=1
    # depthwisecbrm3=true,cbrmbeforedense=1,cat=1效果最好→cbrmbeforedense省略，初始训练精度从0.97下降到0.6，但3-4轮后，训练和验证精度都上升0.98左右------------最终模型精度不看初始的？
    # 只在最后两个depthwise进行cbrm,cbrmbeoforedense=1,初始训练精度从0.97降低至0.93，很快持平;cbrmbeoforedense=0,曲线变化小
    # depthwisecbrm1=1,cbrmbeforedense=0,cat=1,精度和depthwise2cbrm2=1,cbrmbeforedense=0,cat=1差不多

    '''
    2024-1-31-11-50 测试
    训练集0.3 3207-6995
    depthwisecbrm,not low features x high attention scores
    depthwisecbrm1=1,cbrmbefordense=0,cat=1,初始训练精度0.93，初始验证0.97
    depthwisecbrm1=1,cbrmbefordense=0,cat=0,初始训练0.82，初始验证0.94




    
    '''

    '''
    1.depthwise中attention几层对精度影响不大，只有一层的影响是初始训练精度低，对初始验证精度几乎没有影响；但是如果没有，对精度影响较大
    2.cbrmbeforedense对精度影响也不大,没有影响是初始训练精度低，对初始验证精度几乎没有影响
    3.cat影响较大，cat由true-false,对初始验证精度有影响，从0.98-0.4
    4.目前看depthwise的注意力机制差距不大：低水平特征相乘高水平分数和直接高水平特征x高水平分数
    5.所有的模块都没有，只有cat,cat false 优于 cat true；cat false初始训练精度0.86,初始验证0.94
    
    最终选择的模块. depthwise最后一个模块的注意力机制【必须】+cbrmbenfordense=1【可选】+cat=1【必须】
    
    '''
    for dropout in [0]:
        for l2 in [0]:
            for concat in [False]:
          
                    # train( modeltag,depthwisecbrm=0,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=0,concat=concat,dropout=dropout,l2=l2,agu=agu)
                    # train( modeltag,depthwisecbrm=0,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=1,concat=concat,dropout=dropout,l2=l2,agu=agu)
                    # train( modeltag,depthwisecbrm=0,depthwisecorrd=0,ecabeforedense=1,cbrmbeforedense=0,concat=concat,dropout=dropout,l2=l2,agu=agu)
                    # train(modeltag, depthwisecbrm=0,depthwisecorrd=1,ecabeforedense=0,cbrmbeforedense=0,concat=concat,dropout=dropout,l2=l2,agu=agu)
                    # train( modeltag,depthwisecbrm=0,depthwisecorrd=1,ecabeforedense=0,cbrmbeforedense=1,concat=concat,dropout=dropout,l2=l2,agu=agu)
                    # train( modeltag,depthwisecbrm=0,depthwisecorrd=1,ecabeforedense=1,cbrmbeforedense=0,concat=concat,dropout=dropout,l2=l2,agu=agu)
                    # train( modeltag,depthwisecbrm=1,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=0,concat=concat,dropout=dropout,l2=l2,agu=agu)
                    # train( modeltag,depthwisecbrm=1,depthwisecorrd=0,ecabeforedense=1,cbrmbeforedense=0,concat=concat,dropout=dropout,l2=l2,agu=agu)
                   
                    # train(modeltag, depthwisecbrm=0,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=0,concat=True,dropout=dropout,l2=l2,agu=agu,weight=weight)
                    # train( modeltag,depthwisecbrm=0,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=1,concat=True,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    # train(modeltag, depthwisecbrm=1,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=1,concat=concat,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=True)

                    # train( modeltag,depthwisecbrm=0,cbrmlowhigh=0,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=0,concat=False,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    # train( modeltag,depthwisecbrm=0,cbrmlowhigh=0,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=0,concat=True,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    # train( modeltag,depthwisecbrm=0,cbrmlowhigh=0,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=1,concat=True,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    # train( modeltag,depthwisecbrm=1,cbrmlowhigh=0,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=1,concat=True,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    # train( modeltag,depthwisecbrm=1,cbrmlowhigh=0,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=0,concat=True,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    # train( modeltag,depthwisecbrm=0,cbrmlowhigh=0,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=0,concat=False,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    train( modeltag,depthwisecbrm=0,cbrmlowhigh=0,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=0,concat=0
                          ,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                   
                   
                    # train( modeltag,depthwisecbrm=1,cbrmlowhigh=1,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=0,concat=True,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    # train( modeltag,depthwisecbrm=1,cbrmlowhigh=1,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=1,concat=True,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    # train( modeltag,depthwisecbrm=1,cbrmlowhigh=1,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=0,concat=False,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    # train( modeltag,depthwisecbrm=1,cbrmlowhigh=1,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=1,concat=False,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    # train( modeltag,depthwisecbrm=1,cbrmlowhigh=1,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=1,concat=True,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    
                   
                
                    # train( modeltag,depthwisecbrm=1,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=0,concat=concat,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)


                    # train( modeltag,depthwisecbrm=0,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=0,concat=concat,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    # train( modeltag,depthwisecbrm=0,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=0,concat=True,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    
                  
                    
                    
                    # train( modeltag,depthwisecbrm=1,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=0,concat=True,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    # train( modeltag,depthwisecbrm=1,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=0,concat=concat,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)

                    
                #     train(modeltag, depthwisecbrm=0,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=0,concat=concat,dropout=dropout,l2=l2,agu=agu)

                # # for depthwisecrbrm in [0,1]:
                #     for depthwisecorrd in [1,0]:
                #         if depthwisecbrm==0 and depthwisecorrd==1:
                #             continue
                #         for ecabeforedense in [0,1]:
                #             for cbrmbeforedense in [1,0]:
                #                 if ecabeforedense==1 and cbrmbefordense==1:
                #                     continue
                #                 for concat in [1,0]:
                #                     train( depthwisecbrm,depthwisecorrd,ecabeforedense,cbrmbeforedense,concat,dropout,l2,agu)
            

                 
             
 






    

  


    
  
   
    
    
    
    
    
    
    
    
    
    # xtrain,ytrain=np.load(savetrainxPath),np.load(savetrainyPath) #loadandprocessdata(savetrainxPath,savetrainyPath)
    # xval,yval= np.load(savevalxPath),np.load(savevalyPath) #loadandprocessdata(savevalxPath,savevalyPath)
    # print(np.unique(ytrain,return_counts=True),np.unique(yval,return_counts=True))
    # # xtrain,ytrain=random.shuffle(xtrain,ytrain)
    # # xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.3, random_state=42)
    # ytrain=np.where(ytrain!=0,1,ytrain)
    # ytrain= tf.convert_to_tensor(ytrain, dtype=tf.float32)
    # ytrain=to_categorical(ytrain, num_classes=2)
    # yval=np.where(yval!=0,1,yval)
    # yval= tf.convert_to_tensor(yval, dtype=tf.float32)
    # yval=to_categorical(yval, num_classes=2)
    # x=np.concatenate([xtrain,xval],axis=0)
    # y=np.concatenate([ytrain,yval],axis=0)
    # print(y.shape)
    # mean=np.nanmean(x,axis=(0,1,2))
    # std=np.nanstd(x,axis=(0,1,2))
    # xtrain=np.where(np.isnan(xtrain), 0, (xtrain - mean) / std)
    # xval=np.where(np.isnan(xval), 0, (xval- mean) / std)
    # epochs=20
    # h,w,c=xtrain.shape[1],xtrain.shape[2],xtrain.shape[-1]
    # model=medianCnn2d1(inputshape=(h,w,c),channelratio=8,onlybands=False,dropout=0,L2=0)#inputshape,channelratio,bandswithindex=False,bandposition=14,indexposition=17,sar=False,dropout=0.2,L2=0
   
    # model.compile(optimizer=optimizer ,loss= cross_loss, metrics=['accuracy'])
    # print(model.summary())
    # # model.compile(optimizer=optimizer ,loss= 'mse', metrics='mae')
    # # loss='mse', metrics=['mae']
    # start=time.time()
    # compose_transform = Compose([RandomRotation(),RandomVerticalFlip(),RandomHorizontalFlip()])#RandomContrast(),transformer.RandomBrightness()
    # mytrain=dataagument(xtrain,ytrain,64)
    # steps_per_epoch=len(xtrain)/64

    # if xval.shape[1]==1:
    #     xval=np.squeeze(xval,axis=1)
    # history = model.fit_generator(mytrain, steps_per_epoch=steps_per_epoch,validation_data=(xval,yval),
    #             epochs=epochs,callbacks=callback_,verbose=2,shuffle=True)
    # history=model.fit(xtrain,ytrain,epochs=15,validation_data=(xval, yval),
    # batch_size= batch_size,callbacks=callback_,verbose=2,shuffle=True)
    # model=tf.keras.models.load_model(saveModelPath,custom_objects={"K": K})
    # if xtest.shape[1]==1:
    #     xtest=np.squeeze(xtest,axis=1)
    # a= model.evaluate(xtest, ytest)
    # print('accuracy',a)
    # accuracy()
    
    # plot(history)
    # Create a TensorFlow dataset
    # x1 = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))

    # Apply the transformations using the map function
    # train_dataset = x1.map(apply_transforms)
    # Apply the RandomRotation instance to the dataset using the map function
    # xtrain,ytrain=RandomRotationAll()(xtrain, ytrain)
    
    # 
    # print(train_dataset[0].shape,train_dataset[1].shape) #'MapDataset' object is not subscriptable
    # print(xval.shape,yval.shape)
    # Transpose xval
    # xval = np.transpose(xval, (0, 2, 3, 1, 4))  # Corrected dimensions
    # Additional steps
    # batch_size = 64
    # shuffle_buffer_size = 10000
    # train_dataset = train_dataset.batch(batch_size)
    # # train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer_size)
    # # train_dataset = train_dataset.repeat(num_epochs)
    # train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # print( train_dataset,' train_dataset')
    # # Assuming train_dataset is a batched dataset
    # for batch_x, batch_y in train_dataset.take(1):
    #     print("Input shape:", batch_x.shape)
    #     print("Label shape:", batch_y.shape)

    # # Reshape xval
    # xval= np.reshape(xval, (xval.shape[0], xval.shape[1], xval.shape[2], xval.shape[3] * xval.shape[4]))
   
    # myval=val_data_generator(xval,yval,64)
    # print(mytrain,myval)
    # x =mytrain.next() #: 'generator' object has no attribute 'next'
    # y =myval.next()
    # for batch in myval:
    #     print(batch[0].shape,batch[1].shape) # image (32,256,256,3)  label (32,256,256,2)
# just check some 
    # for i in range(10):
    #     print(mytrain.__next__()[0].shape, mytrain.__next__()[1].shape)

   
#%%
 
        # add(3,6)
    # a=4
    # # %%
    # history = model.fit_generator(mytrain, steps_per_epoch=steps_per_epoch,validation_data=(xval,yval),
    #             epochs=10,callbacks=callback_,verbose=2,shuffle=True)
    # # history=model.fit(train_dataset,epochs=15,validation_data=(xval, yval),
    # # batch_size= batch_size,callbacks=callback_,verbose=2,shuf7*9fle=True)
    # model=tf.keras.models.load_model(saveModelPath,custom_objects={"K": K})
    # model.summary()
    # a= model.evaluate(xtest, ytest)
    # print(a)
    # accuracy()
    # plot(history)

    # history = model.fit_generator(mytrain, steps_per_epoch=steps_per_epoch,validation_data=(xval,yval),
    #             epochs=10,callbacks=callback_,verbose=2,shuffle=True)
    # # history=model.fit(train_dataset,epochs=15,validation_data=(xval, yval),
    # # batch_size= batch_size,callbacks=callback_,verbose=2,shuffle=True)
    # model=tf.keras.models.load_model(saveModelPath,custom_objects={"K": K})
    # plot(history)