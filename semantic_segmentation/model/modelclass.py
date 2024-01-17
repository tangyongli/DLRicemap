 
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
# from semantic_segmentation.dataset.loaddataset import DataLoader



class DataLoader:
    '''
   input x shape is sample,times,h,w,channels
   channels sequences:b2 vvmin vhmin vhmax vvmax b3 b4 b8 b5 b6 b7 b8a b11 b12 lswimin evimax ndvimax  lat lon 19bands
   Attributes:
        trainx (str): input x path (default no norminiation).
        trainy (str): input y path (default no onehot encode).
        patch_size (int):  (default is 11).
        include_index (bool): whether iclude_index into training(default is False).
        bandposition (int): Position of the lastband in the data (default is 14).the 14th bands is the last band
        indexposition (int): Position of the index in the data (default is 17).
    '''
    def __init__(self, trainx, trainy, patch_size=11,include_index=False,bandposition=14,indexposition=17):
        self.trainx = trainx
        self.trainy = trainy
        self.patch_size = patch_size
        self.include_index =include_index
        self.index=indexposition
        self.band=bandposition
    def norminiation(self,x,y,extractposition):
        
        t,h,w, channels = x.shape[1],x.shape[2], x.shape[3],x.shape[-1]
        if len(y.shape)==1:
            y = tf.convert_to_tensor(y, dtype=tf.float32)
            y = to_categorical(y, num_classes=2)
     
        x_bands = x[..., :, :, 0:extractposition]
        x = np.transpose(x_bands, (0, 2, 3, 1, 4))
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        mean = np.nanmean(x, axis=(0, 1, 2))
        std = np.nanstd(x, axis=(0, 1, 2))
        x_train = np.where(np.isnan(x), 0, (x - mean) / std)
        np.save(rf'D:\ricemodify\dataset\s1s2medianmaxmincloudmask\xmyyuanmean{x.shape[0]}x{x.shape[1]}x{x.shape[2]}x{x.shape[3]}.npy', mean)
        np.save(rf'D:\ricemodify\dataset\s1s2medianmaxmincloudmask\xmyyuanstd{x.shape[0]}x{x.shape[1]}x{x.shape[2]}x{x.shape[3]}.npy', std)

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],t, -1)
        x = np.transpose(x_train, (0, 3, 1, 2, 4))
        print(x_train.shape)

        ratio = np.sum(x_train == 0) / x_train.size
        print(f"值为0的比率: {ratio:.2%}")
        return x,y
    def __call__(self):
        x = np.load(self.trainx)
        y = np.load(self.trainy)
        if self.include_index:
            x,y=self.norminiation(x,y,self.index)
            print(x.shape,y.shape)
        else:
            x,y=self.norminiation(x,y,self.band)
            print('xxxxx',x.shape,y.shape)
       
        return x, y

# class DatasplitandAugment(DataLoader):
    # def __init__(self,x,y,include_index=False):
    #      super().__init__(x, y, include_index=include_index)
def DatasplitAugment(x,y):
        xtrain, X_temp, ytrain, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
        xval, xtest, yval, ytest = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
        xtrain=np.transpose(xtrain,(0,2,3,1,4))
        print('xy',xtrain.shape,xval.shape)
        xtrain = np.reshape(xtrain, (xtrain.shape[0] , xtrain.shape[1], xtrain.shape[2],xtrain.shape[3]*xtrain.shape[4]))
        mytrain=dataagument(xtrain,ytrain,64)
        steps_per_epoch=len(xtrain)/64
        if xval.shape[1]==1:
            xval=np.squeeze(xval,axis=1)
        if xtest.shape[1]==1:
            xtest=np.squeeze(xtest,axis=1)
        return mytrain,xval,yval,xtest,ytest, steps_per_epoch

class DatasplitAugment():
        def __init__(self,agu):
            self.agu=True

        def datasplit(self,x,y):
            xtrain, X_temp, ytrain, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
            xval, xtest, yval, ytest = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
            xtrain=np.transpose(xtrain,(0,2,3,1,4))
            print('xy',xtrain.shape,xval.shape)
            xtrain = np.reshape(xtrain, (xtrain.shape[0] , xtrain.shape[1], xtrain.shape[2],xtrain.shape[3]*xtrain.shape[4]))
            if self.agu==True:
                mytrain=dataagument(xtrain,ytrain,64)
                steps_per_epoch=len(xtrain)/64
                if xval.shape[1]==1:
                    xval=np.squeeze(xval,axis=1)
                if xtest.shape[1]==1:
                    xtest=np.squeeze(xtest,axis=1)
                return mytrain,xval,yval,xtest,ytest, steps_per_epoch
            else:
                if xtrain.shape[1]==1:
                    xtrain=np.squeeze(xtrain,axis=1)
                if xval.shape[1]==1:
                    xval=np.squeeze(xval,axis=1)
                if xtest.shape[1]==1:
                    xtest=np.squeeze(xtest,axis=1)
                return xtrain,ytrain,xval,yval,xtest,ytest

class DataAndModelLoader(DataLoader):
    '''
   父类中的参数 super().__init__(x,y,patch_size, include_index,bandposition,indexposition)必须在子类的参数 __init__(self,x,y,patch_size,include_index,bandposition,indexposition,inputtimes=1,channelratio=8)
   中包括；子类可以增加新的参数，或者修改父类的参数
    
    '''
    def __init__(self,x,y,patch_size,include_index,bandposition,indexposition,inputtimes=1,channelratio=8):#自己类的属性
        super().__init__(x,y,patch_size, include_index,bandposition,indexposition) #The super().__init__(x, y, ...) call in the constructor of the base class (DataLoader) initializes the base class with these values.;继承类的属性
        self.channelratio=channelratio #前者是变量名称；后者是参数
        self.inputtimes=inputtimes
        # self.x1=x
        # self.x2=y
        self.data_loader = DataLoader(x, y, patch_size, include_index, bandposition, indexposition)#类里面传递的是真实的参数值，so you cannot use x1 and x2,they are variabvle
        self.bandposition=bandposition
        self.indexposition=indexposition
      

    def load_data_and_train_model(self):
        x,y=self.data_loader()
        print(x.shape,y.shape)
        mytrain, xval, yval, xtest, ytest, steps_per_epoch =DatasplitAugment(True).datasplit(x,y)#DatasplitandAugment(x,y)()
        inputbands=xval.shape[-1]
        from semantic_segmentation.model.models import  medianCnn2d
        model = medianCnn2d(inputshape=(self.patch_size, self.patch_size, inputbands), channelratio=self.channelratio, 
        bandswithindex=self.include_index,bandposition=self.bandposition,indexposition=self.indexposition,sar=True)
        model.compile(optimizer=optimizer ,loss= cross_loss, metrics=['accuracy'])
        # steps_per_epoch=len(mytrain)/64
        print('ddddddddddddddddddddddddddddddd',xval.shape, yval.shape, xtest.shape, ytest.shape)
        # Train the model with the loaded data
        model=model.fit_generator(mytrain, steps_per_epoch=steps_per_epoch, validation_data=(xval, yval),
                                 epochs=20, callbacks=callback_, verbose=2, shuffle=True)
        return model,xtest, ytest

       
# Example Usage
savetrainxPath =r'D:\ricemodify\dataset\s1s2medianmaxmincloudmask\xmyyuan10915x1x11x11x19.npy'#D:\ricemodify\dataset\withcloud\datasplit\xnomeanstd10915x2x11x11x15.npy',r'D:\ricemodify\dataset\withcloud\datasplit\ynomeanstd10915.npy'#D:\ricemodify\dataset\datasplit\xnomeanstd12473x2x33x33x15.npy',r'D:\ricemodify\dataset\datasplit\ynomeanstd12473.npy'
savetrainyPath = r'D:\ricemodify\dataset\s1s2medianmaxmincloudmask\ymyyuan10915x1x11x11x19.npy'
print('111111111111111111111111111111111111111111111111')
# 创建一个物体data_and_model_loader = DataAndModelLoader(x=savetrainxPath, y=savetrainyPath,patch_size=11,include_index=False,bandposition=14,indexposition=17)
# # 调用方法
# data_and_model_loader.load_data_and_train_model()

# 