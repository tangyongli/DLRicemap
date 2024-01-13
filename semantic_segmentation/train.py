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
from cfgs import *
from semantic_segmentation.dataset.transformer_ImageDataGenerator import dataagument,val_data_generator#augment_data
from semantic_segmentation.dataset import transformer
from semantic_segmentation.dataset.transformer import  Compose,RandomRotation,RandomContrast,RandomScale,RandomHorizontalFlip,RandomVerticalFlip
from keras.utils import to_categorical
# from semantic_segmentation.dataset.datagenerate import xtrain,xval,xtest,ytrain,yval,ytest
# from semantic_segmentation.dataset.datagenerateclass import xpath
# print(xpath)
def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(999)
   random.seed(999)
   np.random.seed(999)
   tf.random.set_seed(999)
reset_random_seeds()
#%%
def plot(history):
    plt.figure(figsize=(12, 4))
    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.suptitle(saveVersion)
    plt.tight_layout()
    # plt.show() # 渲染了图像可以保存，但是保存的是空白，正确的顺序是不要渲染
    plt.savefig(plotPath_) #plt.savefig(...) will then create an additional figure which might be why you end up with an open figure in the end.
    plt.show()
    time.sleep(5)
    # # Clear the current axes.
    # plt.cla() 
    # # Clear the current figure.
    # plt.clf() 
    # # Closes all the figure windows.
    # plt.close('all')   
    # # plt.close(fig)
    # # gc.collect()
    # # plt.close() 


def accuracy():
    loss,accuracy= model.evaluate(xtest, ytest)
    pred=model.predict(xtest)
    maxpred=np.argmax(pred,axis=-1)
    maxpred.shape
    ytest1=np.argmax(ytest,axis=-1)
    conf_mat = confusion_matrix(ytest1, maxpred)
    # Plot confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(maxpred), yticklabels=np.unique(maxpred))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(saveVersion)
    plt.text(conf_mat.shape[1] + 0.4, conf_mat.shape[0] - 1.2, f'Loss: {loss:.2f}', ha='left', va='center', color='red')
    plt.text(conf_mat.shape[1] + 0.4, conf_mat.shape[0] - 1, f'Accuracy: {accuracy:.2f}', ha='left', va='center', color='red')
    # plt.text(enddoy, evi_value, f'({enddoy}, {end_evi_value:.2f})', ha='left', va='top')
    # plt.show()
    plt.legend(loc='upper left')
    plt.savefig(confusion_matrixpath)
    plt.show()
    time.sleep(5)
    print('plt',plt)
    # plt.cla() 
    # # Clear the current figure.
    # plt.clf() 
    # # Closes all the figure windows.
    # plt.close('all')   

def loadandprocessdata(trainx,trainy,patchsize=11,includeindex=True):
    x=np.load(trainx)
    y=np.load(trainy)
    # x[np.isnan(x)]=0
    # ratio = np.sum(x== 0) / x.size #6376*49*121
    # print(f"值为0的比率: {ratio:.2%}") #值为-1的比率: 1.61%'
    
    print(x.shape)
    patchshape,channels=x.shape[2],x.shape[-1]
    print(patchshape,channels)
    x=x[...,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,0:channels] # sample,2,11,11,15
    # print('x',x.shape)
    y= tf.convert_to_tensor(y, dtype=tf.float32)
    y=to_categorical(y, num_classes=2)
    if includeindex:
        xbands=x[...,:,:,0:12]
        print(x.shape)
        xtimegeo=x[...,:,:,12:15]
        x=np.transpose(xbands,(0,2,3,1,4))
        x=x.reshape(x.shape[0],x.shape[1],x.shape[2],-1) #xtrain
        mean=np.nanmean(x,axis=(0,1,2))
        std=np.nanstd(x,axis=(0,1,2))
        xtrain = np.where(np.isnan(x), 0, (x - mean) / std)
        # xtrain=(x-mean)/std
        np.save(rf'D:\ricemodify\dataset\datasplit\xmyyuanstd{x.shape[0]}x{x.shape[1]}x{x.shape[2]}x{x.shape[3]}.npy',std)
        np.save(rf'D:\ricemodify\dataset\datasplit\xmyyuanmean{x.shape[0]}x{x.shape[1]}x{x.shape[2]}x{x.shape[3]}.npy',mean)

        # xtrain=(x-mean)/std
        # xtrain[np.isnan(xtrain)]=0
        xtrain=xtrain.reshape(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],times,bandswithindex)
        xtrain=np.transpose(xtrain,(0,3,1,2,4))
        print(xtrain.shape)
       
        ratio = np.sum(xtrain== 0) / xtrain.size #6376*49*121
        print(f"值为0的比率: {ratio:.2%}") #值为-1的比率: 1.61%'
        x=np.concatenate([xtrain,xtimegeo],axis=-1)
        # print('xconcat',x.shape)
        xtrainbands=x[...,0:12]
        time1=x[...,12:13]/365
        time2=2*(time1-0.5)
        geo1=x[...,13:14]/90
        geo2=x[...,14:15]/180
        x=np.concatenate([xtrainbands,time1,time2,geo1,geo2],axis=-1)
        x[np.isnan(x)]=0
        ratio = np.sum(x== 0) / x.size #6376*49*121
        print(f"值为0的比率: {ratio:.2%}") #值为-1的比率: 1.61%'
        
    else:
        xbands=x[...,:,:,3:12]
        xtimegeo=x[...,:,:,12:15]
        print('xbands',xbands.shape)
        x=np.transpose(xbands,(0,2,3,1,4))
        x=x.reshape(x.shape[0],x.shape[1],x.shape[2],times*bandswithoutindex) #xtrain
        mean=np.nanmean(x,axis=(0,1,2))
        std=np.nanstd(x,axis=(0,1,2))
        xtrain = np.where(np.isnan(x), 0, (x - mean) / std)
        np.save(rf'D:\ricemodify\dataset\datasplit\xmyyuanstd{x.shape[0]}x{x.shape[1]}x{x.shape[2]}x{x.shape[3]}.npy',std)
        np.save(rf'D:\ricemodify\dataset\datasplit\xmyyuanmean{x.shape[0]}x{x.shape[1]}x{x.shape[2]}x{x.shape[3]}.npy',mean)
        # xtrain=(x-mean)/std
        xtrain=xtrain.reshape(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],times,-1)
        xtrain=np.transpose(xtrain,(0,3,1,2,4))
        print(xtrain.shape)
      
        x=np.concatenate([xtrain,xtimegeo],axis=-1)
        xtrainbands=x[...,0:9]
        time1=x[...,9:10]/365
        time2=2*(time1-0.5)
        geo1=x[...,10:11]/90
        geo2=x[...,11:12]/180
        x=np.concatenate([xtrainbands,time1,time2,geo1,geo2],axis=-1)
        x[np.isnan(x)]=0
        ratio = np.sum(x== 0) / x.size #6376*49*121
        print(f"值为0的比率: {ratio:.2%}") #值为-1的比率: 1.61%'
    return x,y

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--inputshape", type=str)
    # parser.add_argument("--numFilters", type=str)
    # parser.add_argument("--dropout", type=float)
    # parser.add_argument("--learning_rate", type=float)
    # parser.add_argument("--batch_size", type=int)
    # parser.add_argument("--optimizer", type=str)
    # args = parser.parse_args()
   
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    savetrainxPath,savetrainyPath=r'D:\ricemodify\dataset\withcloud\datasplit\xnomeanstd10915x2x11x11x15.npy',r'D:\ricemodify\dataset\withcloud\datasplit\ynomeanstd10915.npy'#D:\ricemodify\dataset\datasplit\xnomeanstd12473x2x33x33x15.npy',r'D:\ricemodify\dataset\datasplit\ynomeanstd12473.npy'
    x,y=loadandprocessdata(savetrainxPath,savetrainyPath,patchsize=11,includeindex=False)
    print('x',x.shape)

    # random.shuffle(x)     

   

    from semantic_segmentation.model.models import DualCnn2dGeotimeCbrm
    model=DualCnn2dGeotimeCbrm(inputshape,channelratio=4,dual=True,bandswithindex=False,dropout=0,L2=0.002) # ratio越小参数越大
    def metric_func( y_pred, y_true):
        y_pred = tf.where(y_pred < 0.5, tf.zeros_like(y_pred, dtype=tf.float32),tf.ones_like(y_pred, dtype=tf.float32))
        acc = tf.reduce_mean(1 - tf.abs(y_true - y_pred))
        return acc
    model.compile(optimizer=optimizer ,loss= cross_loss, metrics=['accuracy'])
    # print(model.summary())
    # model.compile(optimizer=optimizer ,loss= 'mse', metrics='mae')
    # loss='mse', metrics=['mae']
    start=time.time()
    # Splitting the data into training (70%), validation (20%), and test (10%) sets
    # y=np.argmax(y, axis=-1)
    xtrain, X_temp, ytrain, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
    xval, xtest, yval, ytest = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)


    # ytrain,yval, ytest=  np.argmax(ytrain, axis=-1),np.argmax(yval, axis=-1), np.argmax(ytest, axis=-1)
    # maskval=(yval == 0)
    # xval=xval[maskval]
    # yval=yval[maskval]
    # masktest=(ytest == 0)
    # xtest=xtest[masktest]
    # ytest=ytest[masktest]
    # 找到标签为0的位置
    # masktrain = (ytrain == 0)
    # # 使用掩码提取对应的值
    # xtrain= xtrain[masktrain]
    # ytrain= ytrain[masktrain]
    # xpreo=np.random.random((5000,11,11,32))
    # ypreo=np.ones((5000,))
    xtrain=np.transpose(xtrain,(0,2,3,1,4))
    print('xy',xtrain.shape,xval.shape)
    xtrain = np.reshape(xtrain, (xtrain.shape[0] , xtrain.shape[1], xtrain.shape[2],xtrain.shape[3]*xtrain.shape[4]))
    # xtrain=np.concatenate((xtrain,xpreo),axis=0)
    # ytrain=np.concatenate((ytrain,ypreo),axis=0)
    # Shuffle the combined arrays
    # combined_data = list(zip(xtrain, ytrain))
    # np.random.shuffle(combined_data)
    # # # Unpack the shuffled data
    # value_shuffled, label_shuffled = zip(*combined_data)
    # xtrain = np.array(value_shuffled)
    # ytrain = np.array(label_shuffled)

    # x1= tf.data.Dataset.from_tensor_slices((xtrain, y))
    # train_dataset= x1.map(lambda x,y: transformer.Compose([
           
    #         transformer.RandomRotation,
    #         transformer. RandomContrast,
    #         ])(x,y)) # map x y should retu

    compose_transform = Compose([RandomRotation(),RandomVerticalFlip(),RandomHorizontalFlip()])#RandomContrast(),transformer.RandomBrightness()

    # Define a function for mapping in Dataset.map
    def apply_transforms(x, y):
        x_transformed, y_transformed = compose_transform(x, y)
        x_transformed=tf.reshape(x_transformed, (inputheight,inputwidth,times,-1))
        x_transformed=tf.transpose(x_transformed,(2,0,1,3))
        return x_transformed, y_transformed
    mytrain=dataagument(xtrain,ytrain,64)
    steps_per_epoch=len(xtrain)/64
    history = model.fit_generator(mytrain, steps_per_epoch=steps_per_epoch,validation_data=(xval,yval),
                epochs=15,callbacks=callback_,verbose=2,shuffle=True)
    # history=model.fit(train_dataset,epochs=15,validation_data=(xval, yval),
    # batch_size= batch_size,callbacks=callback_,verbose=2,shuffle=True)
    model=tf.keras.models.load_model(saveModelPath,custom_objects={"K": K})
    a= model.evaluate(xtest, ytest)
    print(a)
    accuracy()
    plot(history)
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
    # a= model.evaluate(xtest, ytest)
    # print(a)
    # accuracy()
    # plot(history)