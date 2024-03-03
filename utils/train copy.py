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
from RF_DL.model.module import *
# from model.models import singlebranch
from keras.layers import Input, SeparableConv2D, DepthwiseConv2D, GlobalAveragePooling1D
from RF_DL.plot import *
import logging
import inspect
# def reset_random_seeds():
#    os.environ['PYTHONHASHSEED']=str(999)
#    random.seed(999)
#    np.random.seed(999)
#    tf.random.set_seed(999)
# reset_random_seeds()


def medianCnn2d(inputshape,channelratio, inputtag,depthwisecbrm,depthwisecorrd,ecabeforedense,cbrmbeforedense,concat,dropout,L2):
    if inputtag==0:
        startbands,endbands=0,39
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
    x=depthwiseattention(x1,channelratio=channelratio,depthwisecbrm=depthwisecbrm,depthwisecorrd=depthwisecorrd)
    x=layers.Conv2D(256,(3,3),strides=(2,2),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x= keras.layers.ReLU()(x)
    x = layers.MaxPooling2D(strides=2, padding="same")(x)
    x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    if ecabeforedense==True:
         x=eca_block(x, b=1, gama=2)
    if cbrmbeforedense==True:
        x=channel_attention(x, 8) #inputs,geoinputs,ratio=8,geo=True,dense=False
        x=spatial_attention(x)
    x=layers.GlobalAveragePooling2D()(x)
    x=Flatten()(x)
    x=Dense(256,kernel_regularizer=reg)(x)
    if concat==True:
        xcenter=layers.Lambda(lambda x: x[...,inputheight//2:inputheight//2+1,inputwidth//2:inputwidth//2+1,0:inputschannels])(x1)
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
def dataprogress(savetrainxPath, savetrainyPath, savevalxPath, savevalyPath):
    xtrain,xval=np.load(savetrainxPath),np.load(savevalxPath)
    patchshape=xtrain.shape[1]
    channels=xtrain.shape[-1]
    xtrain=xtrain[...,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,0:channels]
    xval=xval[...,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,0:channels]
    
   
    # xval, xtest, yval, ytest = train_test_split(X_temp, y_temp, test_size=6/8, random_state=42)
    ytrain=np.load(savetrainyPath) #loadandprocessdata(savetrainxPath,savetrainyPath)
    yval= np.load(savevalyPath) #loadandprocessdata(savevalxPath,savevalyPath)
    xtrain, X_temp, ytrain, y_temp = train_test_split(xtrain, ytrain, test_size=0.7, random_state=42)
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
   


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()

    savetrainxPath=r"D:\ricemodify\datasetRF\2022patchsize11\medianq75max\trainxgrid61324nomeanstd_10708x11x11x39.npy"#"D:\ricemodify\datasetRF\pathsize11\traingrid61324\new2022trainxgrid61324nomeanstd_10708x11x11x9.npy"
    savetrainyPath=  r"D:\ricemodify\datasetRF\2022patchsize11\medianq75max\trainygrid61324_10708x11x11x5.npy"#"D:\ricemodify\datasetRF\pathsize11\traingrid61324\new2022trainygrid61324_10708x11x11x2.npy"
    savevalxPath= r"D:\ricemodify\datasetRF\2022patchsize11\medianq75max\valxgrid1729nomeanstd_7031x11x11x39.npy"#"D:\ricemodify\datasetRF\pathsize11\valgrid1729\new2023valxgrid1729nomeanstd_7031x11x11x9.npy"
    savevalyPath= r"D:\ricemodify\datasetRF\2022patchsize11\medianq75max\valygrid1729_7031x11x11x5.npy"#"D:\ricemodify\datasetRF\pathsize11\valgrid1729\new2023valygrid1729_7031x11x11x2.npy"
 
    # 数据选择
    # 0:meadianq75max 1:median 2:Q75 3:max
    patchsize=11
    year=2022
    inputtag='medianq75max'
    # savemeanstdtag=f'{year}_{inputtag}(10bands3index)'
    savemeanstddir=rf"D:\ricemodify\datasetRF\{year}patchsize{patchsize}\{inputtag}"
    
    if inputtag=='medianq75max':
        inputtype=0
     
    if inputtag=='median':
          inputtype=1
     
    if inputtag=='q75':
          inputtype=2
     
    if inputtag=='max':
          inputtype=3
  
    xtrain,xval,ytrain,yval,meanpath,stdpath,traincounts,valcounts= dataprogress(savetrainxPath, savetrainyPath, savevalxPath, savevalyPath)
    # 超参数
    batch_size=64
    learning_rate=0.0001
    epochs=20
    plateauPatience=5
    dropout,l2=0,0

    # 定义指数衰减函数
    def lr_schedule(epoch, lr):
        decay_factor = 0.95  # 每个epoch学习率衰减的因子
        if epoch % 10 == 0 and epoch > 0:
            return lr * decay_factor
        return lr
    
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    cross_loss=keras.losses.CategoricalCrossentropy(from_logits=False)
    cosineloss=tf.keras.losses.CosineSimilarity(axis=1)
    # 模型结构
    depthwisecbrm=True
    ecabeforedense=False
    cbrmbefordense=False
    depthwisecorrd=False
    concat=False
    # 损失函数
    loss=cross_loss
    h,w,c=xtrain.shape[1],xtrain.shape[2],xtrain.shape[-1]
    start=time.time()
    def train( modeltag,depthwisecbrm,depthwisecorrd,ecabeforedense,cbrmbeforedense,concat,dropout,l2,agu):
        start=time.time()
        if modeltag=='depthwise':
            model=medianCnn2d(inputshape=(h,w,c),channelratio=8,inputtag=inputtype,depthwisecbrm=depthwisecbrm,depthwisecorrd=depthwisecorrd,ecabeforedense=ecabeforedense,cbrmbeforedense=cbrmbeforedense,concat=concat,dropout=dropout,L2=l2)#inputshape,channelratio,bandswithindex=False,bandposition=14,indexposition=17,sar=False,dropout=0.2,L2=0simplecnn2d(inputshape=(h,w,c),num_filters=[32,64,128,256,256],dropratio=0)
            model.compile(optimizer=optimizer ,loss= loss, metrics=['accuracy'])
            start=time.time()
            # log
            saveVersion=f'{inputtag}(13)train{traincounts}val{valcounts}_agu{agu}_depthwisecbrm{depthwisecbrm}_depthwisecorrd{depthwisecorrd}__ecabfdense{ecabeforedense}_cbrmbfdense{cbrmbeforedense}__cat{concat}_drop{dropout}_l2{l2}' #centerpixelcatcenter
        if modeltag=='simplecnn2d':
            numfilters=[32,64,128,256,256]
            model=simplecnn2d(inputshape=(h,w,c),num_filters=numfilters,dropratio=0)
            model.compile(optimizer=optimizer ,loss= loss, metrics=['accuracy'])
            layername='x'.join(map(str, numfilters))
            cbrm,coord,eca=0,0,0
            saveVersion=f'{modeltag}_{layername}_cbrm{cbrm}_coord{coord}_eca{eca}'
        
        workEnv= 'D:/ricemodify/runRFDL/train'
        checkpointDir = os.path.join(workEnv, f'{year}patchsize{patchsize}/log/{saveVersion}') #/train/model is wrong
        plotdir=os.path.join(workEnv, f'{year}patchsize{patchsize}/plot')
        resumeModelDir= os.path.join(checkpointDir, 'resume')
        os.makedirs(checkpointDir, exist_ok=True) # os.makedirs(logsDir, exist_ok=True)
        os.makedirs(resumeModelDir, exist_ok=True)
        os.makedirs(plotdir,exist_ok=True)
        plotpath=os.path.join(plotdir,f'{saveVersion}.jpg')
        resumeModelpath=os.path.join(resumeModelDir, f'{saveVersion}.h5')
        saveModelPath = os.path.join(checkpointDir,  f'{saveVersion}.h5')
        print('saveModelPath',saveModelPath)
        # plotPath = os.path.join(checkpointDir, f'{savemeanstdtag}_lossaccuracy.png')
        confusion_matrixpath=os.path.join(checkpointDir, f'{saveVersion}_confusion_matrix.png')
        callback_ = [
                    callbacks.ModelCheckpoint(monitor='val_accuracy', filepath=saveModelPath, mode='max',
                                            save_best_only=True, save_weights_only=False, verbose=1),
                    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=plateauPatience,
                                                min_delta=5e-3, min_lr=1e-6, verbose=1),
                    callbacks.CSVLogger(filename=os.path.join( checkpointDir ,  saveVersion+ '.csv')),
                    # TensorBoard(log_dir=logsDir_,histogram_freq=1)
            ]
  
        if agu:
            mytrain=dataagument(xtrain,ytrain,64)
            steps_per_epoch=len(xtrain)/64
            history = model.fit_generator(mytrain, steps_per_epoch=steps_per_epoch,validation_data=(xval,yval),
                    epochs=epochs,callbacks=callback_,verbose=2,shuffle=True)
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
        logging.getLogger().setLevel(logging.INFO)
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
        if modeltag=='simplecnn2d':
            model_function_source = inspect.getsource(simplecnn2d)
            print(model_function_source)
            logging.info('Model Function Source Code:\n%s',  model_function_source)
        if modeltag=='depthwise':
            model_function_source = inspect.getsource(medianCnn2d)
            print(model_function_source)
            logging.info('Model Function Source Code:\n%s',  model_function_source)
      
        # logging.info('Model Structure:')
        # model.summary(print_fn=lambda x: logging.info(x))
        # print(model.summary())
        # model.compile(optimizer=optimizer ,loss= 'mse', metrics='mae')
        # loss='mse', metrics=['mae']
        compose_transform = Compose([RandomRotation(),RandomVerticalFlip(),RandomHorizontalFlip()])#RandomContrast(),transformer.RandomBrightness()
        # Log training results
  

        plot(history,saveVersion,plotpath)
    depthwisecbrm=0
    ecabeforedense=0
    cbrmbeforedense=0
    depthwisecorrd=0
    concat=0
    agu=0
    modeltag='depthwise'#'simplecnn2d' #
    train(modeltag,depthwisecbrm,depthwisecorrd,ecabeforedense,cbrmbeforedense,concat,dropout,l2,agu)
    # for agu in [False]:
    #     for dropout in [0.2,0.3,0.5]:
    #         for l2 in [0,0.0001,0.001]:
    #             for depthwisecrbrm in [True,False]:
    #                 for depthwisecorrd in [True,False]:
    #                     if depthwisecbrm==True and depthwisecorrd==True:
    #                         continue
    #                     for ecabeforedense in [False,True]:
    #                         for cbrmbeforedense in [True,False]:
    #                             if ecabeforedense==True and cbrmbefordense==True:
    #                                 continue
    #                             for concat in [True,False]:
    #                                 train( depthwisecbrm,depthwisecorrd,ecabeforedense,cbrmbeforedense,concat,dropout,l2,agu)
            

                 
             
 






    

  


    
  
   
    
    
    
    
    
    
    
    
    
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