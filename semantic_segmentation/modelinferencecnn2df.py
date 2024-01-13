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
from cfgs import saveMeanPath,saveStdPath,predictworkEnv_
               
#%%
#r"D:\ricemodify\run\train\patchsize11\model\sample12474imggeneraterotatflip_dual1resnetattention64mpx128mpx128twox128cnn1dxcat64128_128128_geodate4.h5"
# img_dir=r'D:\ricemodify\dataset\origin_img\quxian'
# mean=np.load(saveMeanPath)
# std=np.load(saveStdPath)
# print(saveMeanPath)
# print(mean.shape,std.shape)
def load_tiles(folder,loadrow,loadcol,saveMeanPath,saveStdPath): # 返回的shape统一是高度、宽度、6monthsx11channels
    imgtwo=[]
    file=[os.path.join(folder,file)  for file in os.listdir(folder) if file.endswith("tif") ]
    img=rio.open(file[0])
    trans=img.transform 
    img=img.read()
    c,h,w=img.shape[0],img.shape[1],img.shape[2]
    print(c,h,w)
    # imgcomposite=np.zeros((h,w,c*2),dtype=np.float32)
    # print(imgcomposite.shape)
    print(saveMeanPath)
    mean=np.load(saveMeanPath)
    print('meanmean',mean.shape)
    std=np.load(saveStdPath)
    for f in file:
            img1=rio.open(f).read()[...,0:loadrow,0:loadcol]
            img1=np.transpose(img1,(1,2,0))
            imgtwo.append(img1)  
    imgtwo=np.array(imgtwo)
    print( imgtwo.shape)  # 2,h,w,15
    imggeo=imgtwo[...,12:15]
    print('imggeo',imggeo.shape)
    time1bands=imgtwo[0:1,:,:,0:12]
    time2bands=imgtwo[1:2,:,:,0:12]
    timebothbands=np.concatenate([time1bands,time2bands],axis=-1) 
    timebothbands=np.squeeze(timebothbands,axis=(0,)) 
    img=(timebothbands-mean)/std
    img= img.reshape(img.shape[0],img.shape[1],2,12)
    img=np.transpose(img,(2,0,1,3))


    time1=imggeo[...,0:1]/365
    time2=2*(time1-0.5)
    geo1=imggeo[...,1:2]/90
    geo2=imggeo[...,2:3]/180
    print('geo1',geo1.shape)
    img=np.concatenate([img,time1,time2,geo1,geo2],axis=-1)
    # img=np.concatenate([img,imggeo],axis=-1)
    print('bothshape',img.shape)
    return img,trans
  
#%%
start=time.time()
'''
以下代码加载图像的开始行，开始列只能从0，0开始
 img1=rio.open(f).read()[...,0:500,0:500]

'''
def predict(model,img_dir,loadrow,loadcol,patchsize,startrow,startcol,endrow,endcol,batchsize,saveVersion,saveMeanPath,saveStdPath): # height,width is the no padding image
    # height,width=img.shape[1]-10,img.shape[2]-10
    results=[] #np.zeros(width,height)
    resultsnot=[]
    patches=[]
    patchcount=0
    margin=patchsize//2
    # img=load_img(img_dir)
    img,trans=load_tiles(img_dir,loadrow,loadcol,saveMeanPath,saveStdPath)    
    for row in range(margin+startrow,endrow+margin,1):
        for col in range(margin+startcol,endcol+margin,1):
            patch=img[:,row-margin:row-margin+patchsize,col-margin:col-margin+patchsize,:]
            # print('patch',patch.shape)
            patches.append(patch) #AttributeError: 'numpy.ndarray' object has no attribute 'append'
            patchcount+=1
            if patchcount==batchsize: #分批预测，每批有batchsize个patch
                patches=np.array(patches) # (512,6, 11,11, 10)
                pred=model.predict(patches) #.reshape(-1,2)#(16,2)
                result=np.argmax(pred,axis=1)
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
                print('resultsnot==0')
                resultsnot==0
            else:         
            # remaining_patches=remaining_patches.append(np.array(patches))# 将未处理的 patch 添加到列表中
            # print(remaining_patches.shape)
    # if remaining_patches:
                # 转换为 NumPy 数组
                patches = np.array(patches)

                # 批量预测剩余的 patches
                pred = model.predict(patches)
                print('pred',pred.shape)
                # print("preremain",pred.shape)
                resultsnot.append(np.argmax(pred, axis=1))
        
    prebatch,prebatchnot=np.array(results),np.array(resultsnot)   
    print(prebatch)
    print(prebatchnot.shape) # (1,64)
    prebatchshape=(prebatch.shape[0])*(prebatch.shape[1])      
    prebatch=prebatch.reshape( prebatchshape,-1)
    print('prebatchshape',prebatch.shape) #(39936, 1)
    prebatchnotshape=(prebatchnot.shape[0])*(prebatchnot.shape[1])  #（64，1）    
    prebatchnot=prebatchnot.reshape( prebatchnotshape,-1)
    print('prebatchnotshape',prebatchnot.shape)
    finalpred=np.append(prebatch,prebatchnot,axis=0)
    finalpred=finalpred.reshape((endrow-startrow,endcol-startcol))
    if img_dir.endswith('quxian'):
        predictarray=os.path.join(predictworkEnv_, f'quxian/patchsize{patchsize}/array')
        predictjpg=os.path.join(predictworkEnv_, f'quxian/patchsize{patchsize}/jpg')
        os.makedirs(predictarray, exist_ok=True)
        os.makedirs(predictjpg, exist_ok=True)
    if img_dir.endswith('linshui'):
        predictarray=os.path.join(predictworkEnv_, f'linshui/patchsize{patchsize}/array')
        predictjpg=os.path.join(predictworkEnv_, f'linshui/patchsize{patchsize}/jpg')

        os.makedirs(predictarray, exist_ok=True)
        os.makedirs(predictjpg, exist_ok=True)
    if img_dir.endswith('yuechi'):
        predictarray=os.path.join(predictworkEnv_, f'yuechi/patchsize{patchsize}/array')
        predictjpg=os.path.join(predictworkEnv_, f'yuechi/patchsize{patchsize}/jpg')
        os.makedirs(predictarray, exist_ok=True)
        os.makedirs(predictjpg, exist_ok=True)

    savepath=os.path.join(predictarray,f'{saveVersion}_{startrow}x{startcol}-{endrow}x{endcol}.npy')
    np.save(savepath,finalpred)
    plt.imshow(finalpred)
    plt.title(saveVersion)
    plt.savefig(os.path.join(predictjpg,f'{saveVersion}_{startrow}x{startcol}-{endrow}x{endcol}.jpg'))
    # plt.show()
    return finalpred

# startrow,startcol,endrow,endcol=2500,2500,3500,3500

#%%
# prebatch.shape
# #%%
# prebatch,pred1=predict(model,img,256*4,11,startrow,startcol,endrow,endcol)
# end=time.time()
# print("推理耗费时间:",end-start)  
# shapes1=(prebatch.shape[0])*(prebatch.shape[1])
# # if len(pred1.shape)==0:
# #     shapes2=(pred1.shape[0])
# # else:
# shapes2=(pred1.shape[0])*(pred1.shape[1])
# prebatch=prebatch.reshape(shapes1,-1)
# # np.save(f"cnn2d7x7x121-3003to4003batch.npy",prebatch)
# pred1=pred1.reshape(shapes2,-1)
# finalpred=np.append(prebatch,pred1,axis=0)
# savepath=os.path.join(predictarray,f'{saveVersion}_{startrow}x{startcol}-{endrow}x{endcol}.npy')#os.path.join(predictarray,saveVersion)

# #%%
# # 最终预测得到的array
# finalpred=finalpred.reshape((endrow-startrow,endcol-startcol))
# np.save(savepath,finalpred)
# %%

# %%
def convertarraytolabeltif(model,patchsize,loadrow,loadcol,startrow,startcol,endrow,endcol,img_dir,orgintif,batchsize,saveVersion,saveMeanPath,saveStdPath):
        labelarray=predict(model,img_dir,loadrow,loadcol,patchsize,startrow,startcol,endrow,endcol,batchsize,saveVersion,saveMeanPath,saveStdPath)
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
        if img_dir.endswith('quxian'):
            predicttif=os.path.join(predictworkEnv_, f'quxian/patchsize{patchsize}/tif')
            os.makedirs(predicttif, exist_ok=True)
        if img_dir.endswith('linshui'):
            predicttif=os.path.join(predictworkEnv_, f'linshui/patchsize{patchsize}/tif')
            os.makedirs(predicttif, exist_ok=True)
        if img_dir.endswith('yuechi'):
            predicttif=os.path.join(predictworkEnv_, f'yuechi/patchsize{patchsize}/tif')
        savetifpath=os.path.join(predicttif,f'{saveVersion}_{startrow}x{startcol}-{endrow}x{endcol}.tif')
        with rio.open(savetifpath, 'w', driver='GTiff', width=width, height=height, count=1, dtype=np.uint8, 
                        crs=tif.crs, transform=dst_transform) as dst:
            dst.write(labelarray, 1)

# orgintif=r"D:\RICEFIELD\dataset\download\quxian\2022quxianlabel0rice1othercropveurban2water_othercropveurban3-8monthmeanstd.tif"

# finalpred=np.load(r'D:\ricemodify\run\dualimagescnn2d745_2500to4500.npy')
# finalpred=finalpred.reshape((1000,1000))
# convertarraytolabeltif(orgintif,finalpred,savetifpath)
# %%
