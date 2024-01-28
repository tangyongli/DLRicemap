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
def load_tiles( folder,mean,std,startrow,startcol,endrow,endcol):
        imgbands = []
        # files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith("tif")]
        files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith("tif") and file[0] in ['1', '2', '3','4','5','6','7']]
        img = rio.open(files[0])
        trans = img.transform 
        # img = img.read()[5:14,...]
        # print(img.shape)
        for f in files:
            if f.endswith("tif"):
                print('imagepath is ',f)
             
                print('f',f)
                f=os.path.join(folder, f)
                img1 = rio.open(f).read()[..., startrow:endrow,startcol:endcol]
                # print('img',img1.shape)
                img1 = np.transpose(img1, (1, 2, 0))
                print(img1.shape)
                imgbands.append(img1)
        bands=np.array(imgbands)
        bands=np.transpose(bands,(1,2,0,3))
        img=np.reshape(bands,(bands.shape[0],bands.shape[1],-1))

        img=(img-mean)/std
        img[np.isnan(img)]=0
        # print(glcm.shape,bands.shape)
        return img, trans
  

def predict(modelpath,batchsize,patchsize,startrow,startcol,endrow,endcol): # height,width is the no padding image  
    model=tf.keras.models.load_model(modelpath,custom_objects={"K": K})
    print(model.summary())
    results=[] #np.zeros(width,height)
    resultsnot=[]
    patches=[]
    patchcount=0
    margin=patchsize//2
    i=0
    # 这样定义意味着加载进来的图像区域都要预测完
    startpredictrow=0
    startpredictcol=0
    endpredictrow=endrow-startrow-patchsize+1
    endpredictcol=endcol-startcol-patchsize+1
    # img=load_img(img_dir)
    for row in range(margin+startpredictrow,endpredictrow+margin,1): #(2000,2000,3000,3000) 预测的endrow ,endcol要在startrow,startcol上减去patchsize，再加上1
        for col in range(margin+startpredictcol,endpredictcol+margin,1):  # endcol:endcol+7----col-6:col+1
            patch=img[row-margin:row-margin+patchsize,col-margin:col-margin+patchsize,:]
            
            if patch.shape != (patchsize,patchsize, img.shape[-1]):
                   i=i+1
                   print(row,col,patch.shape)
                    # 调整形状并填充为-1
                   patch = np.full((patchsize,patchsize, img.shape[-1]), -1)
          
            patches.append(patch) #AttributeError: 'numpy.ndarray' object has no attribute 'append'
            patchcount+=1
            if patchcount==batchsize:
                patches=np.array(patches) # (512,6, 11,11, 10
                pred=model.predict(patches) #.reshape(-1,2)#(16,2)
                result=np.argmax(pred,axis=1)
                # print('result',result.shape)
                results.append(result)
                patchcount=0
                patches=[]
    print(patchcount,i) #4
    if patchcount!=batchsize:
            if patchcount==0:
                resultsnot==0
            else:         
                patches = np.array(patches)
                # 批量预测剩余的 patches
                pred = model.predict(patches)
                resultsnot.append(np.argmax(pred, axis=1))
                        
    return np.array(results),np.array(resultsnot)
def predresult(batchsize,patchsize,startrow,startcol,endrow,endcol,predictdir):

    prebatch,pred1=predict(modelpath,batchsize,patchsize,startrow,startcol,endrow,endcol) #
  
    shapes1=(prebatch.shape[0])*(prebatch.shape[1])
    shapes2=(pred1.shape[0])*(pred1.shape[1])
    prebatch=prebatch.reshape(shapes1,-1)
    pred1=pred1.reshape(shapes2,-1)
    finalpred=np.append(prebatch,pred1,axis=0)
    os.makedirs(predictdir,exist_ok=True)
    finalpred=finalpred.reshape((endrow-startrow-patchsize+1,endcol-startcol-patchsize+1))
    np.save(os.path.join(predicttifdir,saveVersion+f'{startrow}-{endrow}x{startcol}-{endcol}.npy'),finalpred)
    # np.save(os.path.join(predictarray,f"cnnbest3dattentionmeanstdwrongfullyuechi"),finalpred)
    value,count=np.unique(finalpred,return_counts=True)
    print(finalpred.shape,value,count)
    plt.imshow(finalpred)
    plt.title(f'{saveVersion}-{startrow}-{endrow}x{startcol}-{endcol}')
    os.makedirs(predictdir,exist_ok=True)
    savejpg=os.path.join(predictdir,saveVersion+f'{startrow}-{endrow}x{startcol}-{endcol}.jpg')
    plt.savefig(savejpg)
    # plt.show()
    return finalpred

def convertarraytolabeltif(orgintif,labelarray,patchsize,startrow,startcol,endrow,endcol,savetifpath):
        tif=rio.open(orgintif)
        transform=rio.open(orgintif).transform
        rowtop=startrow+patchsize//2
        colleft=startcol+patchsize//2
        rowbottom=endrow-patchsize//2
        colright=endcol-patchsize//2
        left, bottom= transform * (colleft, rowbottom)
        right,top= transform*(colright, rowtop)
        dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs=tif.crs,
        dst_crs=tif.crs,
        width=colright-colleft,
        height=rowbottom-rowtop,
        left=left,
        bottom=bottom,
        right=right,
        top=top
        )
        print( 'dst',dst_width, dst_height) #995 883
        os.makedirs(savetifpath,exist_ok=True)
        outpath=os.path.join(savetifpath,saveVersion+f"{startrow}-{endrow}x{startcol}-{endcol}.tif") #{saveVersion_}_{startrow}-{endrow}x{startcol}-{endcol}
        # outpath=os.path.join(outtifpath,f"{startrow}-{endrow}x{startcol}-{endcol}.tif")
        print(' outpath', outpath)
        with rio.open(outpath, 'w', driver='GTiff', width=dst_width, height=dst_height, count=1, dtype=np.uint8, 
                        crs=tif.crs, transform=dst_transform) as dst:
            dst.write(labelarray, 1)
        print('arraytotifffinshed!!!!')
        return 0

if __name__ == "__main__":

    batchsize=256*5
    inputtag='sevenper'
    year,patchsize= 2022,11#modeldir.split('\\')[-1]
    modeldir=rf"D:\ricemodify\runRFDL\train\{year}{inputtag}patchsize{patchsize}\log"
    # patchsizep=modeldir.split('\\')[-1]
    # import re
    # # Use a regular expression to extract numeric part,正则搜索的是一个字符串，即使是数字
    # patchsize = int(re.search(r'\d+', patchsizep).group())
    # print(year,patchsize)
    arealist=['yuechi','linshui','quxian']
    for area in arealist:
        if area=='yuechi':
            img_dir=r"D:\ricemodify\datasetRF\origin\yuechi2022"
            img_list = [file for file in os.listdir(img_dir) if file.endswith('.tif')]
            orgintif=os.path.join(img_dir,img_list[0])
            startrow,startcol,endrow,endcol=2500,2500,3500,3500
        if area=='linshui':
            img_dir=r"D:\ricemodify\datasetRF\origin\linshui2022"
            img_list = [file for file in os.listdir(img_dir) if file.endswith('.tif')]
            orgintif=os.path.join(img_dir,img_list[0])
            startrow,startcol,endrow,endcol=3000,3500,4000,4500
        if area=='quxian':
            img_dir=r"D:\ricemodify\datasetRF\origin\quxian2022"
            img_list = [file for file in os.listdir(img_dir) if file.endswith('.tif')]
            orgintif=os.path.join(img_dir,img_list[0])
            startrow,startcol,endrow,endcol=3000,1000,4000,2000
            # "D:\ricemodify\runRFDL\train\patchsize11\2022\2022_medianq75max(10bands3index)_depthwisecenterpixelcatcenter"
        for modeltag in os.listdir(modeldir):
            if modeltag=='sevenper(13)train3207val6995_aguFalse_depthwisecbrm1_depthwisecorrd0__ecabfdense0_cbrmbfdense1__catFalse_drop0_l20':
            # if modeltag[0] in ['1', '2', '3','4','5','6']:
                saveVersion=modeltag
                modelpath=os.path.join(modeldir,modeltag)
                print('modelpath is',modelpath)

                logpath=[os.path.join(modelpath, f) for f in os.listdir(modelpath) if f.endswith('.txt')][0]
                modelpath=[os.path.join(modelpath, f) for f in os.listdir(modelpath) if f.endswith('.h5')][0] # [3 1 2] 3是返回值
                with open(logpath, 'r') as log_file:
                    log_lines = log_file.readlines()
                    savextrainpath_value = None
                    for line in log_lines:
                        if 'saveMeanPath' in line:
                            saveMeanPath= line.split(':')[-1].strip()
                            print('Debug - saveMeanPath:', repr(saveMeanPath)) 
                            if saveMeanPath=='':
                                saveMeanPath=rf"D:\ricemodify\datasetRF\{year}{inputtag}patchsize{patchsize}\medianq75max\mean39_2022_medianq75max(10bands3index).npy"
                            mean=np.load(saveMeanPath)
                        if 'saveStdPath' in line:
                            saveStdPath= line.split(':')[-1].strip()
                            if saveStdPath=='':
                                saveStdPath=rf"D:\ricemodify\datasetRF\{year}{inputtag}patchsize{patchsize}\std39_ 2022_medianq75max(10bands3index).npy"
                            std=np.load(saveStdPath)
                    
                img,trans=load_tiles(img_dir,mean,std,startrow,startcol,endrow,endcol)
                predictdir=rf'D:\ricemodify\runRFDL\predict\{area}\{year}{inputtag}patchsize{patchsize}\arraypng\{saveVersion}'
                predicttifdir=rf'D:\ricemodify\runRFDL\predict\{area}\{year}{inputtag}patchsize{patchsize}\tifpath'
                os.makedirs(predictdir,exist_ok=True)
                os.makedirs(predicttifdir,exist_ok=True)
                modeltag='simplecnn2d'#'depthwise
                # savetag=f'medianq75max(13){patchsize}{modeltag}'
                log_file_path=os.path.join(predictdir,saveVersion+'.txt')
                import logging
                import inspect
                # Configure logging
                logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file_path),
                    logging.StreamHandler()  # To also print logs to console
                ]
                )

            
                logging.getLogger().setLevel(logging.INFO)
            
                logging.info('saveMeanPath: %s', saveMeanPath)
                logging.info('saveStdPath: %s', saveStdPath)
                logging.info('saveMOdelPath:%s', modelpath)
            # os.makedirs(predicttifpath,exist_ok=True)
            # os.makedirs(predictjpgpath,exist_ok=True)
            # savepredarraypath,savepredjpgpath,savetifpath=os.path.join(predictarray,saveVersion),os.path.join(predictjpg,saveVersion),os.path.join(predicttif,saveVersion)
                labelarray= predresult(batchsize,patchsize,startrow,startcol,endrow,endcol,predictdir)
                # labelarray=np.load(os.path.join(predictdir,f'{saveVersion}_{startrow}-{endrow}x{startcol}-{endcol}.npy'))
                # predresult(batchsize,patchsize,startrow,startcol,endrow,endcol,predictdir)
                # predict(modelpath,img,batchsize,patchsize,startrow,startcol,endrow,endcol)
                convertarraytolabeltif(orgintif,labelarray,patchsize,startrow,startcol,endrow,endcol,predicttifdir)












