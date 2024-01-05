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
modelpath=saveModelPath
img_dir=r'D:\ricemodify\dataset\quxian'
img_list = [file for file in os.listdir(img_dir) if file.endswith('.tif')]
model=tf.keras.models.load_model(modelpath,custom_objects={"K": K})
model.summary()
mean=np.load(saveMeanPath)
std=np.load(saveModelPath)
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
            img1=rio.open(f).read()
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
img,trans=load_tiles(img_dir)      
#%%
start=time.time()
def predict(model,img,batchsize,patchsize,height,width): # height,width is the no padding image
    # height,width=img.shape[1]-10,img.shape[2]-10
    results=[] #np.zeros(width,height)
    resultsnot=[]
    patches=[]
    patchcount=0
    margin=patchsize//2
    # img=load_img(img_dir)
    for row in range(margin+2500,height+margin,1):
        for col in range(margin+2500,width+margin,1):
            patch=img[:,row-margin:row-margin+patchsize,col-margin:col-margin+patchsize,:]
            # print('patch',patch.shape)
            patches.append(patch) #AttributeError: 'numpy.ndarray' object has no attribute 'append'
            patchcount+=1
            if patchcount==batchsize: #分批预测，每批有batchsize个patch
                patches=np.array(patches) # (512,6, 11,11, 10)
                pred=model.predict(patches) #.reshape(-1,2)#(16,2)
                result=np.argmax(pred,axis=1)
                # print('result',result.shape)
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
                # print("preremain",pred.shape)
                resultsnot.append(np.argmax(pred, axis=1))
                        
    return np.array(results),np.array(resultsnot)

height,width=3500,3500# 结束预测的图像高度和宽度
prebatch,pred1=predict(model,img,256*4,11,height,width)
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
np.save(f"dualimagescnn2d745_2500to4500.npy",finalpred)
#%%
# 最终预测得到的array
finalpred=finalpred.reshape((1000,1000))
# %%
plt.show()
plt.imshow(finalpred)
# %%
def convertarraytolabeltif(orgintif,labelarray,outpath):
        tif=rio.open(orgintif)
        transform=rio.open(orgintif).transform
        width=1000 #labelarray.shape[1]
        height=1000#labelarray.shape[0]
        rowbottom=height
        colleft=2500
        rowtop=2500
        colright=3500
        rowbottom=3500
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
orgintif=r'D:\ricemodify\dataset\quxian\quxian_2023sentinel2_maxevi601-901composite15bandswithdoylatlon.tif'
# orgintif=r"D:\RICEFIELD\dataset\download\quxian\2022quxianlabel0rice1othercropveurban2water_othercropveurban3-8monthmeanstd.tif"
outpath=r'D:\ricemodify\run\predict\3dcnn\yuechi\patchsize33'
os.makedirs(outpath,exist_ok=True)
# outpath=os.path.join(outpath,rf"predpadding{modelFlag_}_{time_}x{inputHeight_}x{inputWidth_}x{inputChannel_}_3000to{height}_3000to{width}.tif")
modelFlag_='cnn2d7x7x121'
outpath=os.path.join(outpath,f"singleimages2500to3500.tif")
convertarraytolabeltif(orgintif,finalpred,outpath)
# %%
