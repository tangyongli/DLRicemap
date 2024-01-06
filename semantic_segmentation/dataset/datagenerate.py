#%%
import pandas as pd
import rasterio as rio
from rasterio.windows import Window
import numpy as np
import os
import tensorflow as tf
import keras
import pandas as pd
import os
import random
import json
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
# from utils.bestparams import *
from cfgs import *

#%%
d=99
c=99
def get_c_value():
    c = 99
    return c

def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(999)
   random.seed(999)
   np.random.seed(999)
   tf.random.set_seed(999)
reset_random_seeds()
df=pd.read_csv(samplepath)
df=df[df['area']!=0]
# df['B2'].replace(2,1,inplace=True)
# df['landcover'].value_counts()
df['lon'] = df['.geo'].apply(lambda geo_str: json.loads(geo_str)['coordinates'][0])
df['lat'] = df['.geo'].apply(lambda geo_str: json.loads(geo_str)['coordinates'][1])
pd.set_option('display.float_format', lambda x: '%.15f' % x)
df['imgpatch']=[None] * len(df)
df['landcover'].value_counts()
linshui=df[df['area'] == 1]
quxian=df[df['area'] == 2]
yuechi=df[df['area']==3]
print(yuechi)
#%%
def load_tiles(folder): # 返回的shape统一是高度、宽度、6monthsx11channels
    imgtwo=[]
    file=[os.path.join(folder,file)  for file in os.listdir(folder) if file.endswith("tif") ]
    img=rio.open(file[0])
    trans=img.transform 
    img=img.read()
    c,h,w=img.shape[0],img.shape[1],img.shape[2]
    # imgcomposite=np.zeros((h,w,30),dtype=np.float32)
    # print(imgcomposite.shape)
    for f in file:
            img1=rio.open(f).read()
            img1=np.transpose(img1,(1,2,0))
            imgtwo.append(img1)  
    imgtwo=np.array(imgtwo)
    print( imgtwo.shape)  # 2,h,w,15
    return imgtwo,trans
def patchfrompoint(df,img,trans,patchsize):
    radius=patchsize//2
    for index, row in df.iterrows():
        lon=df["lon"]
        lat=df["lat"]
        # print(index)
        lon, lat = row["lon"], row["lat"]
        col, row = ~trans * (lon, lat)
        patchrowtop, patchrowbottom, patchcolleft, patchcolright = (
            int(row - radius),
            int(row + radius + 1),
            int(col - radius),
            int(col + radius + 1),
        )
        imgpatch=img[:,patchrowtop:patchrowbottom,patchcolleft:patchcolright,:]
        print('before',imgpatch.shape)
        if imgpatch.shape != (2,patchsize,patchsize,15):
            print('after',imgpatch.shape)
            print( patchrowtop, patchrowbottom, patchcolleft, patchcolright)
            continue
            # print('imgpatch',imgpatch)
            # patch.append(imgpatch)
            # imgpatch[np.isnan(imgpatch)] = -1
        df.at[index,"imgpatch"]=imgpatch # 不能使用loc
            # print('df',df["imgpatch"])
    return df,imgpatch.shape
#%%
def datamerge(df,imgyuechipath,imgquxianpath,imglinshuipath,patchsize,normalize=True):
    # dfyuechi,dflinshui,dfquxian=datasplit(samplepath)
    imgquxian,transquxian=load_tiles(imgquxianpath)
    imglinshui,translinshui=load_tiles(imglinshuipath)
    imgyuechi,transyuechi=load_tiles(imgyuechipath)
    patchyuechi,_=patchfrompoint(yuechi,imgyuechi,transyuechi,patchsize)
    patchyuechi=patchyuechi[patchyuechi['imgpatch'].apply(lambda x: x is not None)]
    patchlinshui,_=patchfrompoint(linshui,imglinshui,translinshui,patchsize)
    patchlinshui=patchlinshui[patchlinshui['imgpatch'].apply(lambda x: x is not None)]
    patchquxian,_=patchfrompoint(quxian,imgquxian,transquxian,patchsize)
    patchquxian=patchquxian[patchquxian['imgpatch'].apply(lambda x: x is not None)]
    df=pd.concat([patchyuechi,patchlinshui, patchquxian], ignore_index=True)
    df= df[df['imgpatch'].apply(lambda x: x is not None)]
    return  df#patchyuechi,patchlinshui,patchquxian
patchdf=datamerge(df,imgyuechipath,imgquxianpath,imglinshuipath,11)
#%%
patchdf.shape
#%%
def samplenormin(patchdf,normalization=True):
    if normalization==True:
        patchdf= patchdf[patchdf['imgpatch'].apply(lambda x: x is not None)]
        xtrain=np.array(patchdf['imgpatch'].tolist())[...,:,:,0:12]
        xtimegeo=np.array(patchdf['imgpatch'].tolist())[...,:,:,12:15]
        xtrain=np.transpose(xtrain,(0,2,3,1,4))
        xtrain=xtrain.reshape(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],24) #xtrain
        mean=np.nanmean(xtrain,axis=(0,1,2))
       
        std=np.nanstd(xtrain,axis=(0,1,2))
        xtrain=(xtrain-mean)/std
        # np.save(rf'D:\ricemodify\dataset\datasplit\xmean{xtrain.shape[0]}x{xtrain.shape[1]}x{xtrain.shape[2]}x{xtrain.shape[3]}x{xtrain.shape[4]}.npy',mean)
        # np.save(rf'D:\ricemodify\dataset\datasplit\xstd{xtrain.shape[0]}x{xtrain.shape[1]}x{xtrain.shape[2]}x{xtrain.shape[3]}x{xtrain.shape[4]}.npy',std)
        # np.save(saveMeanPath,mean)
        # np.save(saveStdPath,std)
        xtrain=xtrain.reshape(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],2,12)
        xtrain=np.transpose(xtrain,(0,3,1,2,4))
        xtrain=np.concatenate([xtrain,xtimegeo],axis=-1)
        patchdf['landcover'].replace(2,1,inplace=True)
        print(patchdf['landcover'].unique())
        ytrain=np.array(patchdf['landcover'].tolist())
        ytrain= tf.convert_to_tensor(ytrain, dtype=tf.float32)
        ytrain=to_categorical(ytrain, num_classes=2)
        print(xtrain.shape,ytrain.shape)
        np.save(rf'D:\ricemodify\dataset\datasplit\x{xtrain.shape[0]}x{xtrain.shape[1]}x{xtrain.shape[2]}x{xtrain.shape[3]}x{xtrain.shape[4]}.npy',xtrain)
        np.save(rf'D:\ricemodify\dataset\datasplit\y{ytrain.shape[0]}.npy',ytrain)
        # 检查是否存在None值
        # contains_none = np.any(xtrain == None)
        # if contains_none:
        #     print("数组中存在None值。")
        # else:
        #     print("数组中没有None值。")
    else:
        xtrain=np.array(patchdf['imgpatch'].tolist())
        ytrain=np.array(patchdf['landcover'].tolist())
        ytrain= tf.convert_to_tensor(ytrain, dtype=tf.float32)
        ytrain=to_categorical(ytrain, num_classes=2)
        print(xtrain.shape,ytrain.shape)
        np.save(rf'D:\ricemodify\dataset\datasplit\x{xtrain.shape[0]}x{xtrain.shape[1]}x{xtrain.shape[2]}x{xtrain.shape[3]}x{xtrain.shape[4]}.npy',xtrain)
        np.save(rf'D:\ricemodify\dataset\datasplit\y{ytrain.shape[0]}.npy',ytrain)
        # np.save(r'D:\ricemodify\dataset\datasplit\xvalylq2x11x11x15.npy',xval)
        # np.save(r'D:\ricemodify\dataset\datasplit\yvalylq2x11x11x15.npy',yval)
        # np.save(r'D:\ricemodify\dataset\datasplit\xtestylq2x11x11x15.npy',xtest)
        # np.save(r'D:\ricemodify\dataset\datasplit\ytestylq2x11x11x15.npy',ytest)

    return xtrain,ytrain#,xval,yval,xtest,ytestr

xtrain,ytrain=samplenormin(patchdf,normalization=True)
#%%
xtrain[np.isnan(xtrain)]=0
# xval1[np.isnan(xval1)]=0
# xtest1[np.isnan(xtest1)]=0
# print(xtrain1.shape,ytrain.shape)
ratio = np.sum(xtrain == 0) / xtrain.size #6376*49*121
print(f"值为-1的比率: {ratio:.2%}") #值为-1的比率: 1.61%'
#%%
xtrainbands=xtrain[...,0:12]
time1=xtrain[...,12:13]/365
time2=2*(time1-0.5)
geo1=xtrain[...,13:14]/90
geo2=xtrain[...,14:15]/180
xtrain=np.concatenate([xtrainbands,time1,time2,geo1,geo2],axis=-1)
xtrain.shape
#%%
from sklearn.model_selection import train_test_split
# Splitting the data into training (70%), validation (20%), and test (10%) sets
xtrain, X_temp, ytrain, y_temp = train_test_split(xtrain, ytrain, test_size=0.3, random_state=42)
xval, xtest, yval, ytest = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
print(xtrain.shape,xval.shape)

