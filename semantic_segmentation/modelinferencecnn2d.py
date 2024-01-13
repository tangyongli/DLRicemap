#%%
import rasterio as rio
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from rasterio.warp import calculate_default_transform, reproject
from rasterio.transform import Affine
# from models import cnn3dattention
from keras import backend as K
from cfgs import *
               
#%%
predictarray=r'D:\ricemodify\run\predict\quxian\patchsize11\array'
predicttif=r'D:\ricemodify\run\predict\quxian\patchsize11\tif'
predictjpg=r'D:\ricemodify\run\predict\quxian\patchsize11\jpg'
os.makedirs(predictarray,exist_ok=True)
os.makedirs(predicttif,exist_ok=True)
os.makedirs(predictjpg,exist_ok=True)

modelpath=saveModelPath#r"D:\ricemodify\run\train\patchsize11\model\sample6114_imggeneraterotatflip_dual1dwise33twice_16326496noattentionafterconcat.h5"#"D:\ricemodify\run\train\patchsize11\model\sample6114_imggeneraterotatflip_dual1dwise33twice_16326496noattentionafterconcat.h5"#saveModelPath#r"D:\ricemodify\run\train\patchsize33\model\sample12473imggeneraterotatflip_dual1dwise35twice_16326496geoattentionafter.h5"#saveModelPath #r"D:\ricemodify\run\train\patchsize11\model\sample12474imggeneraterotatflip_dual1resnetattention64mpx128mpx128twox128cnn1dxcat64128_128128_geodate4.h5"
print(modelpath)
img_dir=r'D:\ricemodify\dataset\withcloud\quxian'
def metric_func( y_pred, y_true):
        y_pred = tf.where(y_pred < 0.5, tf.zeros_like(y_pred, dtype=tf.float32),tf.ones_like(y_pred, dtype=tf.float32))
        acc = tf.reduce_mean(1 - tf.abs(y_true - y_pred))
        return acc
img_list = [file for file in os.listdir(img_dir) if file.endswith('.tif')]
model=tf.keras.models.load_model(modelpath,custom_objects={"K": K,'inputheight':inputheight,'inputwidth':inputwidth,'geotimebands':geotimebands})
model.summary()
# saveMeanPath,saveStdPath=r'D:\ricemodify\dataset\datasplit\xnomeanx11x11x24.npy',r'D:\ricemodify\dataset\datasplit\xricestd12473x11x11x24.npy'
mean=np.load(r"D:\ricemodify\dataset\datasplit\xmyyuanmean10915x11x11x18.npy")
std=np.load(r"D:\ricemodify\dataset\datasplit\xmyyuanstd10915x11x11x18.npy")
print(saveMeanPath)
print(mean.shape,std.shape)

def load_tiles(folder): # 返回的shape统一是高度、宽度、6monthsx11channels
    imgtwo=[]
    file=[os.path.join(folder,file)  for file in os.listdir(folder) if file.endswith("tif") ]
    img=rio.open(file[0])
    trans=img.transform 
    img=img.read()
    c,h,w=img.shape[0],img.shape[1],img.shape[2]
    print(c,h,w)
    # imgcomposite=np.zeros((h,w,c*2),dtype=np.float32)
    # print(imgcomposite.shape)
    for f in file:
            img1=rio.open(f).read()[...,0:5000,0:5000]
            
            img1=np.transpose(img1,(1,2,0))
            imgtwo.append(img1)  
    imgtwo=np.array(imgtwo)
    print( imgtwo.shape)  # 2,h,w,15
    imggeo=imgtwo[...,12:15]
    print('imggeo',imggeo.shape)
    time1bands=imgtwo[0:1,:,:,3:12]
    time2bands=imgtwo[1:2,:,:,3:12]
    timebothbands=np.concatenate([time1bands,time2bands],axis=-1) 
    timebothbands=np.squeeze(timebothbands,axis=(0,)) 
    img=(timebothbands-mean)/std
    img= img.reshape(img.shape[0],img.shape[1],2,9)
    print(img.shape)
    img=np.transpose(img,(2,0,1,3))
    # np.save('imgquxian24bands.npy',img)


    time1=imggeo[...,0:1]/365
    time2=2*(time1-0.5)
    geo1=imggeo[...,1:2]/90
    geo2=imggeo[...,2:3]/180
    print('geo1',geo1.shape)
    img=np.concatenate([img,time1,time2,geo1,geo2],axis=-1)
    # img=np.concatenate([img,imggeo],axis=-1)
    print('bothshape',img.shape)
    return img,trans
img,trans=load_tiles(img_dir)      
#%%
start=time.time()
def predict(model,img,batchsize,patchsize,startrow,startcol,endrow,endcol): # height,width is the no padding image
    # height,width=img.shape[1]-10,img.shape[2]-10
    results=[] #np.zeros(width,height)
    resultsnot=[]
    patches=[]
    patchcount=0
    margin=patchsize//2
    # img=load_img(img_dir)
    for row in range(margin+startrow,endrow+margin,1):
        for col in range(margin+startcol,endcol+margin,1):
            patch=img[:,row-margin:row-margin+patchsize,col-margin:col-margin+patchsize,:]
            # print('patch',patch.shape)
            patches.append(patch) #AttributeError: 'numpy.ndarray' object has no attribute 'append'
            patchcount+=1
            if patchcount==batchsize: #分批预测，每批有batchsize个patch
                patches=np.array(patches) # (512,6, 11,11, 10)
                pred=model.predict(patches) #.reshape(-1,2)#(16,2)
                # threshold = 0.5
                # pred = np.where(pred < threshold, 0, 1)
                result=np.argmax(pred,axis=-1)
                print('result',result.shape)
                results.append(result)
                patchcount=0
                patches=[]
                # i+=1
        
            #   while i>3:
            #     break
    print(patchcount) #4
    if patchcount!=batchsize:
            if patchcount==0:
                resultsnot==0
            else:         
            # remaining_patches=remaining_patches.append(np.array(patches))# 将未处理的 patch 添加到列表中
            # print(remaining_patches.shape)
    # if remaining_patches:
                # 转换为 NumPy 数组
                patches = np.array(patches)

                # 批量预测剩余的 patches
                pred = model.predict(patches)
                pred = np.argmax(pred,axis=-1)#np.where(pred < threshold, 0, 1)
                # print("preremain",pred.shape)
                resultsnot.append(pred)
                        
    return np.array(results),np.array(resultsnot)
# startrow,startcol,endrow,endcol=2500,2500,3500,3500
prebatch,pred1=predict(model,img,256*4,patchsize,startrow,startcol,endrow,endcol)
end=time.time()
#%%
prebatch.shape
#%%
print("推理耗费时间:",end-start)  
shapes1=(prebatch.shape[0])*(prebatch.shape[1])
# if len(pred1.shape)==0:
#     shapes2=(pred1.shape[0])
# else:
shapes2=(pred1.shape[0])*(pred1.shape[1])
prebatch=prebatch.reshape(shapes1,-1)
# np.save(f"cnn2d7x7x121-3003to4003batch.npy",prebatch)
pred1=pred1.reshape(shapes2,-1)
finalpred=np.append(prebatch,pred1,axis=0)
savepath=os.path.join(predictarray,f'{saveVersion}_{startrow}x{startcol}-{endrow}x{endcol}.npy')#os.path.join(predictarray,saveVersion)

#%%
# 最终预测得到的array
finalpred=finalpred.reshape((endrow-startrow,endcol-startcol))
np.save(savepath,finalpred)
# %%


plt.imshow(finalpred)
plt.savefig(os.path.join(predictjpg,f'{saveVersion}_{startrow}x{startcol}-{endrow}x{endcol}.jpg'))
plt.show()
# %%
def convertarraytolabeltif(orgintif,labelarray,outpath):
        tif=rio.open(orgintif)
        transform=rio.open(orgintif).transform
        height=endrow-startrow #labelarray.shape[1]
        width=endcol-startcol#labelarray.shape[0]
        rowbottom=height
        colleft=startcol+patchsize//2
        rowtop=startrow+patchsize//2
        colright=endcol+patchsize//2
        rowbottom=endrow+patchsize//2
        left, bottom= transform * (colleft, rowbottom)
        right,top= transform*(colright, rowtop)
        dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs=tif.crs,
        dst_crs=tif.crs,
        width=width, # 这里需要是输出的图像的像素大小
        height=height,
        left=left,
        bottom=bottom,
        right=right,
        top=top
        )
        print(dst_transform, dst_width, dst_height ,tif.crs)
        with rio.open(outpath, 'w', driver='GTiff', width=width, height=height, count=1, dtype=np.uint8, 
                        crs=tif.crs, transform=dst_transform) as dst:
            dst.write(labelarray, 1)

orgintif=r"D:\RICEFIELD\dataset\download\quxian\2022quxianlabel0rice1othercropveurban2water_othercropveurban3-8monthmeanstd.tif"
savetifpath=os.path.join(predicttif,f'{saveVersion}_{startrow}x{startcol}-{endrow}x{endcol}true.tif')
modelFlag_='cnn2d7x7x121'
savepath=os.path.join(predictarray,f'{saveVersion}_{startrow}x{startcol}-{endrow}x{endcol}.npy')
finalpred=np.load(savepath)
# finalpred=finalpred.reshape((1000,1000))
convertarraytolabeltif(orgintif,finalpred,savetifpath)
# %%

def convertarraytolabeltif(orgintif,labelarray,outpath):
        tif=rio.open(orgintif)
        transform=rio.open(orgintif).transform
        height=5000#labelarray.shape[1]
        width=5000#labelarray.shape[0]
        rowbottom=height
        colleft=0
        rowtop=0
        colright=width
        rowbottom=height
        left, bottom= transform * (colleft, rowbottom)
        right,top= transform*(colright, rowtop)
        dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs=tif.crs,
        dst_crs=tif.crs,
        width=width, # 这里需要是输出的图像的像素大小
        height=height,
        left=left,
        bottom=bottom,
        right=right,
        top=top
        )
        channels=24
        print(dst_transform, dst_width, dst_height ,tif.crs)
        with rio.open(outpath, 'w', driver='GTiff', width=width, height=height, count=channels, dtype=np.float32, 
        crs=tif.crs, transform=dst_transform) as dst:
            # dst.write(labelarray, 1)
            for i in range(channels): # 写入多个波段 rasterio写入的顺序是按照通道 高度 宽度 首先要将转换为该格式，不然输出的像素值不对；tf batch_size, height, width, color_channels
                dst.write(labelarray[i,...], i+1)

# orgintif=r"D:\RICEFIELD\dataset\download\quxian\2022quxianlabel0rice1othercropveurban2water_othercropveurban3-8monthmeanstd.tif"

# modelFlag_='cnn2d7x7x121'
# savetifpath=r'D:\ricemodify\dataset\imgquxian24bands1.tif'
# finalpred=np.load(r'D:\ricemodify\imgquxian24bands.npy')
# finalpred=np.reshape(finalpred,(5000,5000,-1))
# print(finalpred.shape)
# # finalpred=finalpred.reshape((1000,1000))
# convertarraytolabeltif(orgintif,finalpred,savetifpath)
# # %%
