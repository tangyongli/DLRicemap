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
from RF_DL.model.model import *
# from model.models import singlebranch
from keras.layers import Input, SeparableConv2D, DepthwiseConv2D, GlobalAveragePooling1D
from RF_DL.model.loss import *

from RF_DL.plot import *
import logging
import inspect
def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(999)
   random.seed(999)
   np.random.seed(999)
   tf.random.set_seed(999)
reset_random_seeds()
from tensorflow.keras import layers
import keras
from tensorflow.keras.layers import Input, Reshape, LayerNormalization, multiply,Dense,Activation,Add,Flatten,Lambda ,Concatenate, Conv1D, Conv2D
from keras import backend as K
import numpy as np
import tensorflow as tf
# random seed for model reproduction


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

def dataprogress(savetrainxPath1, savetrainyPath1,savetrainxPath2, savetrainyPath2, savevalxPath, savevalyPath,median12max3,median12max4,median13max3,median13max4):
    x1,x2=np.load(savetrainxPath1),np.load(savetrainxPath2)
    y1,y2=np.load(savetrainyPath1),np.load(savetrainyPath2)
    xval,yval=np.load(savevalxPath),np.load(savevalyPath)
    xtrain=np.concatenate([x1,x2],axis=0)
    y=np.concatenate([y1,y2],axis=0)
    y=np.where(y!=0,1,y)
    y= tf.convert_to_tensor(y, dtype=tf.float32)
    ytrain=to_categorical(y,num_classes=2)
    yval=np.where(yval!=0,1,yval)
    yval= tf.convert_to_tensor(yval, dtype=tf.float32)
    yval=to_categorical(yval, num_classes=2)
    patchshape=x1.shape[1]
    if median12max3:
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
    print(xval.shape)
    x=np.concatenate([xtrain,xval])
    print(ytrain.shape)
   

    # y=np.concatenate([ytrain,yval],axis=0)
    mean=np.nanmean(x,axis=(0,1,2))
    std=np.nanstd(x,axis=(0,1,2))
    print('mean',mean.shape,'std',std.shape)
    # xtrain, xval, ytrain, yval = train_test_split(x,y, test_size=0.1, random_state=42)
    # xval,xtest,yval,ytest=train_test_split(x,y, test_size=0.1, random_state=42)
    xtrain=np.where(np.isnan(xtrain), 0, (xtrain - mean) / std)
    xval=np.where(np.isnan(xval), 0, (xval- mean) / std)
    meanpath=os.path.join( savemeanstddir,f'sample{xtrain.shape[0]}xrf{xval.shape[0]}_mean{mean.shape[0]}.npy')#os.path.dirname(savetrainxPath)
    stdpath=os.path.join(savemeanstddir,f'sample{xtrain.shape[0]}xrf{xval.shape[0]}_std{std.shape[0]}.npy')
    np.save(meanpath,mean)
    np.save(stdpath,std)
    return xtrain,xval,ytrain,yval,meanpath,stdpath,xtrain.shape[0],xval.shape[0]
   

def train( modeltag,numfilters,sattention111,sattention011 ,multscalesattetion,csattention,multscalesattetion001,noattention,concatdense,concatcnntrue1d,dropout,l2,agu,epochs,weight=True,learning_rate=0.0001):
        start=time.time()
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        if modeltag=='dualseparablecnn2d':
            model=dualsparableCnn2d(inputtag=inputtype,inputshape=(h,w,c),numfilters=numfilters,sattention111=sattention111,sattention011=sattention011,multscalesattetion=multscalesattetion,multscalesattetion001=multscalesattetion001,csattention=csattention,noattention=noattention,concatdense=concatdense,concatcnntrue1d=concatcnntrue1d,dropout=dropout,L2=l2)#inputshape,channelratio,bandswithindex=False,bandposition=14,indexposition=17,sar=False,dropout=0.2,L2=0simplecnn2d(inputshape=(h,w,c),num_filters=[32,64,128,256,256],dropratio=0)
            print(model.summary())#{modeltag}separadd_sat111
            saveVersion=f'samplesaddgeetrain_rfval2classes_t{traincounts}val{valcounts}_depthwise{sattention111}xsat011{sattention011}xmscalesatt{multscalesattetion}xms001{multscalesattetion001}xnoatt{noattention}_catcnn1dtrue{concatcnntrue1d}cbrm1_drop{dropout}_l2{l2}'
            print(saveVersion)
            print('222222222') 
            if weight:
                loss=binary_weighted_cross_entropy
                  # loss=binary_weighted_cross_entropy
                # loss=keras.losses.CategoricalCrossentropy(label_smoothing=0)
            else:
                loss=cross_loss
            
            model.compile(optimizer=optimizer ,loss=loss, metrics=['accuracy'])
            start=time.time()
            from datetime import datetime
         
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
            # current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            current_date =      current_date .replace(' ', '-').replace(':', '-')
            print(current_date)
            print(current_date)
        
        if modeltag=='simplecnn2d':
            numfilters=[32,64,128,256,256]
            model=simplecnn2d(inputshape=(h,w,c),num_filters=numfilters,dropratio=0)
            if weight:
                # loss=binary_weighted_cross_entropy
                loss=keras.losses.CategoricalCrossentropy(label_smoothing=0)
            model.compile(optimizer=optimizer ,loss= loss, metrics=['accuracy'])
            layername='x'.join(map(str, numfilters))
            cbrm,coord,eca=0,0,0
            saveVersion=f'{modeltag}_{layername}_cbrm{cbrm}_coord{coord}_eca{eca}'
        
        workEnv= f'D:/ricemodify/runRFDL/train/{year}{inputtag}_{modeltag}'
        checkpointDir = os.path.join(workEnv, f'log/{saveVersion}')#f'{year}{inputtag}patchsize{patchsize}/log/{saveVersion}') #/train/model is wrong
        plotdir=os.path.join(workEnv, 'plot')#f'{year}{inputtag}patchsize{patchsize}/plot')
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
            from sklearn.utils import class_weight
            #`sample_weight` argument is not supported when using python generator as input.
            history = model.fit(mytrain,validation_data=(xval,yval),steps_per_epoch=steps_per_epoch,
                    epochs=epochs,callbacks=callback_,verbose=2,shuffle=True)

               
            # if weight:
            #     history = model.fit(xtrain,ytrain, validation_data=(xval,yval),batch_size=64,class_weight=compute_class_weights(ytrain),
            #         epochs=epochs,callbacks=callback_,verbose=2,shuffle=True)
            #     model=  tf.keras.models.load_model(saveModelPath,custom_objects={"K": K,'binary_weighted_cross_entropy':binary_weighted_cross_entropy})
        else:
                history = model.fit(xtrain,ytrain, validation_data=(xval,yval),batch_size=32,
                    epochs=epochs,callbacks=callback_,verbose=2,shuffle=True)

        end=time.time()
        total=(end-start)/60

        plot(history,saveVersion,plotpath,plotpath2)
        return 0

if __name__ == "__main__":
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    savetrainxPath1=r"D:\ricemodify\limiteddataset\2022pathsize11\samplesvalue278.npy"
    savetrainxPath2=r"D:\ricemodify\limiteddataset\2022pathsize11\samplesGEEvalue181.npy"

    savetrainyPath1=r"D:\ricemodify\limiteddataset\2022pathsize11\sampleslabel278.npy"
    savetrainyPath2=r"D:\ricemodify\limiteddataset\2022pathsize11\samplesGEElabel181.npy"
    savevalxPath=r"D:\ricemodify\datasetRF\2022patchsize15\sevenper\valxgrid1729nomeanstd_4103x15x15x91.npy"
    savevalyPath=r"D:\ricemodify\datasetRF\2022patchsize15\sevenper\valygrid1729_4103x15x15x5.npy"
    # 数据选择
    # 0:meadianq75max 1:median 2:Q75 3:max
    patchsize=11
    year=2022
    inputtag='12median3max'#'sevenper'#'medianq75max'
    # savemeanstdtag=f'{year}_{inputtag}(10bands3index)'
    savemeanstddir=rf"D:\ricemodify\limiteddataset\{year}{inputtag}patchsize{patchsize}"
    os.makedirs(savemeanstddir,exist_ok=True)
    median12max3,median12max4,median13max3,median13max4=True,False,False,False
    # if inputtag=='medianq75max' or inputtag=='sevenper'or inputtag=='13median3max' or inputtag=='13median4max' or inputtag=='12median3max' or inputtag=='12median4max':
    #     inputtype=0
    xtrain,xval,ytrain,yval,meanpath,stdpath,traincounts,valcounts= dataprogress(savetrainxPath1, savetrainyPath1,savetrainxPath2, savetrainyPath2, savevalxPath, savevalyPath,median12max3,median12max4,median13max3,median13max4)
    # 超参数
    batch_size=64
    print(xtrain.shape,xval.shape,ytrain.shape,yval.shape)

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
    concat=False
    weight=True
    # 损失函数
    loss=cross_loss#binary_weighted_cross_entropy
    h,w,c=xtrain.shape[1],xtrain.shape[2],xtrain.shape[-1]
    start=time.time()
    epochs=30
    concat=None
    agu=1
    weight=True
    modeltag='dualseparablecnn2d'
    inputtype=0

    tf.config.experimental.enable_op_determinism()

    for dropout in [0]:
        for l2 in [0,0.00002,0.0002,0.002]:
            for concat in [False]:
                    # train( modeltag,sattention=0,multscalesattetion=1,multscalesattetion001=0,csattention=0,noattention=0,concat=True,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    train( modeltag,numfilters=3 ,sattention111=1,sattention011=0,multscalesattetion=0,multscalesattetion001=0,csattention=0,noattention=0,concatdense=0,concatcnntrue1d=1,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    # train( modeltag,numfilters=3 ,sattention111=1,sattention011=0,multscalesattetion=0,multscalesattetion001=0,csattention=0,noattention=0,concatdense=1,concatcnntrue1d=0,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    # train( modeltag,depthwisecbrm=1,cbrmlowhigh=1,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=0,concat=True,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                    # train( modeltag,depthwisecbrm=1,cbrmlowhigh=1,depthwisecorrd=0,ecabeforedense=0,cbrmbeforedense=1,concat=True,dropout=dropout,l2=l2,agu=agu,epochs=epochs,weight=weight)
                 

                 
             
        






    

  


    
  
   
    
    
    
    
    
    
    
    
    
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
    # # batch_size= batch_size,callbacks=callback_,verbose=2,shuffle=True)
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