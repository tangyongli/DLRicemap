
import os
import tensorflow as tf
from keras import backend as K
import time
folder11=r'D:\ricemodify\run\train\patchsize11\modeldpwise'
folder33=r'D:\ricemodify\run\train\patchsize33\model'
batchsize11=512
batchsize33=512
loadrow,loadcol=5000,5000
patchsize=11
startrowq,startcolq,endrowq,endcolq=3000,3500,4900,4900#4900,4900#2500,2500,3500,3500
startrowl,startcoll,endrowl,endcoll=3000,3000,4900,4900#4900,4900#2500,2500,3500,3500
startrowy,startcoly,endrowy,endcoly=3000,3000,4900,4900#4900,4900#2500,2500,3500,3500
orgintifquxian=r'D:\ricemodify\dataset\origin_img\quxian\quxian_2023sentinel2_maxevi601-901composite15bandswithdoylatlon.tif'
orgintiflinshui=r'D:\ricemodify\dataset\origin_img\linshui\linshui_2023sentinel2_maxevi601-901composite15bandswithdoylatlon.tif'
orgintifyuechi=r'D:\ricemodify\dataset\origin_img\yuechi\yuechi_2023sentinel2_maxevi601-901composite15bandswithdoylatlon.tif'
img_dirq=r'D:\ricemodify\dataset\origin_img\quxian'
img_dirl=r'D:\ricemodify\dataset\origin_img\linshui'
img_diry=r'D:\ricemodify\dataset\origin_img\yuechi'
saveMeanPath33,saveStdPath33=r'D:\ricemodify\dataset\datasplit\xmean12473x33x33x24.npy',r'D:\ricemodify\dataset\datasplit\xstd12473x33x33x24.npy'
saveMeanPath11,saveStdPath11=r'D:\ricemodify\dataset\datasplit\x12474x11x11x24mean.npy',r'D:\ricemodify\dataset\datasplit\x12474x11x11x24std.npy'
# for patchsize in [11,33]:
#     print(patchsize)
#     if patchsize==11:
for file in os.listdir(folder11):
            start=time.time()
            if file.endswith('h5'):
                f=os.path.join(folder11,file)
                print(f)
                h5 = file.find('.h5')
                if h5 != -1:
                    modelname = file[:h5]
                    print("Extracted modelname:", modelname)
                else:
                    print("Invalid filename format")
                    modelname=file
                # 可选：如果您希望去除可能存在的额外空格
                modelname = modelname.strip()
                saveVersion=modelname
                saveMeanPath,saveStdPath=saveMeanPath11,saveStdPath11
                print("Trimmed modelname:", modelname)
                batchsize=batchsize11
                try:
                    model=tf.keras.models.load_model(f,custom_objects={"K": K})
                 
                    from semantic_segmentation.modelinferencecnn2df import *
                    #model,startrow,startcol,endrow,endcol,img_dir,orgintif,batchsize,saveVersion
                    q=convertarraytolabeltif(model,patchsize,loadrow,loadcol,startrowq,startcolq,endrowq,endcolq,img_dirq,orgintifquxian,batchsize,saveVersion,saveMeanPath,saveStdPath) #注意开始的行和列不要超出加载的image的范围
                    l=convertarraytolabeltif(model,patchsize,loadrow,loadcol,startrowl,startcoll,endrowl,endcoll,img_dirl,orgintiflinshui,batchsize,saveVersion,saveMeanPath,saveStdPath)
                    y=convertarraytolabeltif(model,patchsize,loadrow,loadcol,startrowy,startcoly,endrowy,endcoly,img_diry,orgintifyuechi,batchsize,saveVersion,saveMeanPath,saveStdPath)
                    print(f"Error processing file {file}: {str(e)}")
                    total=time.time()-start
                    print(f'model{modelname}finish predict!')
                    print(f'time minitues:',total/60)
                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
    
    # else:
    #     for file in os.listdir(folder33):
    #         start=time.time()
    #         if file.endswith('h5'):
    #             f=os.path.join(folder33,file)
    #             print(f)
    #             h5 = file.find('.h5')
    #             if h5 != -1:
    #                 modelname = file[:h5]
    #                 print("Extracted modelname:", modelname)
    #             else:
    #                 print("Invalid filename format")
    #                 modelname=file
    #                     # 可选：如果您希望去除可能存在的额外空格
    #             modelname = modelname.strip() 
    #             saveMeanPath,saveStdPath=saveMeanPath33,saveStdPath33
    #             print("Trimmed modelname:", modelname)
    #             batchsize=batchsize33
    #             startrowq,startcolq,endrowq,endcolq=2000,2000,3000,3000#4900,4900#2500,2500,3500,3500
    #             startrowl,startcoll,endrowl,endcoll=2000,2000,3000,3000#4900,4900#2500,2500,3500,3500
    #             startrowy,startcoly,endrowy,endcoly==2000,2000,3000,3000#4900,4900#2500,2500,3500,3500
            # try:
            #     model=tf.keras.models.load_model(f,custom_objects={"K": K})
            #     saveVersion=modelname
            #     from semantic_segmentation.modelinferencecnn2df import *
            #     #model,startrow,startcol,endrow,endcol,img_dir,orgintif,batchsize,saveVersion
            #     q=convertarraytolabeltif(model,patchsize,loadrow,loadcol,startrowq,startcolq,endrowq,endcolq,img_dirq,orgintifquxian,batchsize,saveVersion,saveMeanPath,saveStdPath) #注意开始的行和列不要超出加载的image的范围
            #     l=convertarraytolabeltif(model,patchsize,loadrow,loadcol,startrowl,startcoll,endrowl,endcoll,img_dirl,orgintiflinshui,batchsize,saveVersion,saveMeanPath,saveStdPath)
            #     y=convertarraytolabeltif(model,patchsize,loadrow,loadcol,startrowy,startcoly,endrowy,endcoly,img_diry,orgintifyuechi,batchsize,saveVersion,saveMeanPath,saveStdPath)
            #     print(f"Error processing file {file}: {str(e)}")
            #     total=time.time()-start
            #     print(f'model{modelname}finish predict!')
            #     print(f'time minitues:',total/60)
        
            # except Exception as e:
            #     print(f"Error processing file {file}: {str(e)}")
    
            