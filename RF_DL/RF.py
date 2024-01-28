
import seaborn as sns   
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import os
import numpy as np
import rasterio as rio
import tensorflow as tf
import keras.layers as layers
import keras
input_shape=(33,33,32)
num_filters=[32,64,128,256]

savetrainxPath=r"D:\ricemodify\datasetRF\pathsize11\traingrid61324\new2023trainxgrid61324nomeanstd_10551x11x11x39.npy"#"D:\ricemodify\datasetRF\pathsize11\traingrid61324\trainxgrid61324nomeanstd10593x11x11x39.npy"#"D:\ricemodify\datasetRF\pathsize11\traingrid61324\trainxgrid61324nomeanstd10659.npy"
savetrainyPath= r"D:\ricemodify\datasetRF\pathsize11\traingrid61324\new2023trainygrid61324_10551x11x11x2.npy"#"D:\ricemodify\datasetRF\pathsize11\traingrid61324\trainygrid6132410593x11x11x39x2.npy"#"D:\ricemodify\datasetRF\pathsize11\traingrid61324\trainygrid6132410659.npy"
savevalxPath= r"D:\ricemodify\datasetRF\pathsize11\valgrid1729\new2023valxgrid1729nomeanstd_6965x11x11x39.npy"#"D:\ricemodify\datasetRF\pathsize11\valgrid1729\valxgrid1729nomeanstd7003x11x11x39.npy"#"D:\ricemodify\datasetRF\pathsize11\valgrid1729\valxgrid1729nomeanstd7012.npy"
savevalyPath= r"D:\ricemodify\datasetRF\pathsize11\valgrid1729\new2023valygrid1729_6965x11x11x2.npy"#"D:\ricemodify\datasetRF\pathsize11\valgrid1729\valygrid17297003x11x11x39x2.npy"#"D:\ricemodify\datasetRF\pathsize11\valgrid1729\valygrid17297012.npy"
xtrain,ytrain=np.load(savetrainxPath),np.load(savetrainyPath) #loadandprocessdata(savetrainxPath,savetrainyPath)
xval,yval= np.load(savevalxPath),np.load(savevalyPath) #loadandprocessdata(savevalxPath,savevalyPath)
xtrain=xtrain[...,5:6,5:6,0:13]
xtrain=np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[-1]))
xval=xval[...,5:6,5:6,0:13]
xval=np.reshape(xval,(xval.shape[0],xval.shape[-1]))

trainsample,valsample,bands=xtrain.shape[0],xval.shape[0],13#xtrain.shape[-1]

# # RF模型训练
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
def train_rf_classifier(X_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

rfmodel=train_rf_classifier(xtrain,ytrain)

# 保存模型
from joblib import dump

def evaluate_rf_classifier(rf_classifier, X_test, y_test):
    predictions = rf_classifier.predict(X_test)  #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    accuracy = metrics.accuracy_score(y_test,  predictions)
    print(accuracy,'accuracy') 
    #0.6344410876132931 CNN的特征，随机森林输入                                                                                                
    #0.7734138972809668只用随机森林
    return accuracy
acc=evaluate_rf_classifier(rfmodel,xval,yval)
print(acc) #0.5438066465256798
# 加载模型
#%%
# rf_model = load('only_rf_model_4630train.joblib')
# imgs,labels=np.load('tile2_331_32val_imgrf.npy'),np.load('tile2_331val_labelrf.npy')
# pred = rf_model.predict(imgs)
# accuracy = metrics.accuracy_score(labels,  pred)
# print(accuracy,'accuracy') 
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(labels,pred)
# # 设置 seaborn 风格
# sns.set(font_scale=1.2)
# plt.figure(figsize=(8, 6))
# # 绘制混淆矩阵的热图
# sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
#             xticklabels=['norice', 'rice'], yticklabels=['norice', 'rice'])
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
plt.show()
#0.6344410876132931 CNN的特征，随机森林输入
#0.7734138972809668只用随机森林
## 只使用RF推理
#%%
rfpath=r"D:\RICEFIELD\only_rf_model_4630train.joblib"
import joblib
import numpy as np
import matplotlib.pyplot as plt

img_dir=r'D:\ricemodify\datasetRF\origin\yuechi2022'#D:\ricemodify\dataset\s1s2medianmaxmincloudmask\yuechi"
def load_img(image):
    image=rio.open(image).read()
    image=np.transpose(image,(1,2,0))
    imgband=image[:,:,0:5]/10000
    imgindex=image[:,:,5:8]
    image=np.concatenate((imgband,imgindex),axis=-1) # 33,33,8
    image[np.isnan(image)]=-1
    return(image)

def load_tiles( folder,startrow,startcol,endrow,endcol):
        imgbands = []
        files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith("tif")]
        img = rio.open(files[0])
        trans = img.transform 
        # img = img.read()[5:14,...]
        # print(img.shape)
        for f in os.listdir(folder):
            if f.endswith("tif"):
                # if f.startswith('1'):
                    print('f',f)
                    f=os.path.join(folder, f)
                    img1 = rio.open(f).read()[..., startrow:endrow,startcol:endcol]
                    print('img',img1.shape)
                    img1 = np.transpose(img1, (1, 2, 0))
                    print(img1.shape)
                    imgbands.append(img1)
              
        
        bands=np.array(imgbands)
        img=np.reshape(bands,(bands.shape[1],bands.shape[2],-1))
        print(img.shape)
        # img=(img-mean)/std
        img[np.isnan(img)]=0
        # print(glcm.shape,bands.shape)
        return img, trans
startrow,startcol,endrow,endcol=2000,2000,4000,4000
img,trans=load_tiles(img_dir,startrow,startcol,endrow,endcol)  
imgpred=img.reshape(-1,bands)
pred=rfmodel.predict(imgpred)
predimg=pred.reshape(endrow-startrow,endcol-startcol)
plt.imshow(predimg)
plt.show()
# np.save('rfpredict_3341_3341_tile2.npy',predimg)
# %%
value,count=np.unique(predimg,return_counts=True)
print(value,count)
#%%
from rasterio.warp import calculate_default_transform, reproject
from rasterio.transform import Affine
a=3
orgintif=r"D:\ricemodify\dataset\s1s2medianmaxmincloudmask\yuechi\1yuechi_2023sentinel12cloudmask_medianmaxmincomposite301to901_19bandswithlatlon.tif"
outpath=rf"D:\ricemodify\runRFDL\predict\new2023yuechiRFtrian{trainsample}val{valsample}bands{bands}2000to4000.tif"
def convertarraytolabeltif(orgintif,labelarray,outpath):
        tif=rio.open(orgintif)
        transform=rio.open(orgintif).transform
        widths=labelarray.shape[1]
        heights=labelarray.shape[0]
    
        colleft=startcol
        rowtop=startrow
        rowbottom=heights+rowtop
        colright=widths+  colleft

        left, bottom= transform * (colleft, rowbottom)
        right,top= transform*(colright, rowtop)
        dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs=tif.crs,
        dst_crs=tif.crs,
        width=widths,
        height=heights,
        left=left,
        bottom=bottom,
        right=right,
        top=top
        )
        # os.makedirs(outpath,exist_ok=True)
        with rio.open(outpath, 'w', driver='GTiff', width=widths, height=heights, count=1, dtype=np.uint8, 
                        crs=tif.crs, transform=dst_transform) as dst:
            dst.write(labelarray, 1)
#%%
convertarraytolabeltif(orgintif,predimg,outpath)
#%%
a=3
b=7