import os
import json
import pandas as pd
import numpy as np
import rasterio as rio
xpath=" "
ypath=" "
class PatchDataLoader:
    def __init__(self, sample_path, img_yuechi_path, img_quxian_path, img_linshui_path, patch_size,time_size,band_size):
        self.df = pd.read_csv(sample_path)#.sample(1000,random_state=42)
       
        self.df['imgpatch'] = [None] * len(self.df)
        self.img_yuechi, self.trans_yuechi = self.load_tiles(img_yuechi_path)
        self.img_quxian, self.trans_quxian = self.load_tiles(img_quxian_path)
        self.img_linshui, self.trans_linshui = self.load_tiles(img_linshui_path)
        self.areas = {
            'yuechi': self.df[self.df['area'] == 0],
            'linshui': self.df[self.df['area'] == 1],
            'quxian': self.df[self.df['area'] == 2]
        }

        self.patch_size = patch_size
        self.time_size = time_size
        self.band_size = band_size

    def load_tiles(self, folder):
        img_list = []
        files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith("tif")]
        img = rio.open(files[0])
        trans = img.transform 
        img = img.read()
        for f in files:
            img1 = rio.open(f).read()
            img1 = np.transpose(img1, (1, 2, 0))
            img_list.append(img1)
        img_array = np.array(img_list)
        print(img_array.shape)
        return img_array, trans

    def patch_from_point(self,area, img, trans):
        areadf = self.areas[area]
       
        print('areadf',areadf.head(3))
        areadf['lon'] = areadf['.geo'].apply(lambda geo_str: json.loads(geo_str)['coordinates'][0])
        areadf['lat'] = areadf['.geo'].apply(lambda geo_str: json.loads(geo_str)['coordinates'][1])
        for index, row in areadf.iterrows():
            radius = self.patch_size // 2
            arealon,arealat=row["lon"],row["lat"]
            col, row = ~trans * (arealon, arealat)
            patch_row_top, patch_row_bottom, patch_col_left, patch_col_right = (
                int(row - radius),
                int(row + radius + 1),
                int(col - radius),
                int(col + radius + 1),
            )
            img_patch = img[:, patch_row_top:patch_row_bottom, patch_col_left:patch_col_right, :]
            print(img_patch.shape)
            if img_patch.shape == (self.time_size, self.patch_size, self.patch_size, self.band_size):
                print(img_patch.shape)
                areadf.at[index, "imgpatch"] = img_patch
            else:
                print('imgshapeout',img_patch.shape)
                areadf.at[index, "imgpatch"] = None
            # return areadf # if return is located here,the fucntion will be only run one times,for the first row/
            # return statement terminates the function and returns control to the caller.
        return areadf

    def generate_patches(self):
        yuechi_patchdf = self.patch_from_point('yuechi',  self.img_yuechi, self.trans_yuechi)  # ('yuechi', lat, self.img_yuechi, self.trans_yuechi)
        linshui_patchdf = self.patch_from_point('linshui', self.img_linshui, self.trans_linshui)
        quxian_patchdf = self.patch_from_point('quxian', self.img_quxian, self.trans_quxian)
        print(quxian_patchdf.head(5)) #(3544, 9)
        # print(quxian_patchdf['imgpatch'][0].shape) 
        yuechi_patchdf= yuechi_patchdf[ yuechi_patchdf['imgpatch'].apply(lambda x: x is not None)]
        linshui_patchdf= linshui_patchdf[ linshui_patchdf['imgpatch'].apply(lambda x: x is not None)]
        quxian_patchdf= quxian_patchdf[ quxian_patchdf['imgpatch'].apply(lambda x: x is not None)]
        # print(quxian_patchdf['imgpatch'][0].shape) 
        df1 =pd.concat([yuechi_patchdf,linshui_patchdf, quxian_patchdf], ignore_index=True)
       
        imgtoarray=np.array(df1['imgpatch'].tolist())
        labeltoarray=np.array(df1['landcover'].tolist())
        print( 'imgarray',imgtoarray.shape,labeltoarray.shape)
        yuechi_patchdf,quxian_patchdf,linshui_patchdf=None,None,None
        return imgtoarray,labeltoarray
        

    # def get_data_frame(self):
    #     return self.df


# # Example Usage
class normalization():

    def __init__(self,imgarray):
        self.x=imgarray
        self.time_size=imgarray.shape[0]
        self.bandswithindex=imgarray.shape[-1]
    def meanstd(self):
        xbands=self.x[...,:,:,0:12]
        print(x.shape)
        xtimegeo=x[...,:,:,12:15]
        x=np.transpose(xbands,(0,2,3,1,4))
        x=x.reshape(x.shape[0],x.shape[1],x.shape[2],-1) #xtrain
        mean=np.nanmean(x,axis=(0,1,2))
        std=np.nanstd(x,axis=(0,1,2))
        xtrain=(x-mean)/std
        np.save(rf'D:\ricemodify\dataset\withcloud\datasplit\xmyyuanstd{x.shape[0]}x{x.shape[1]}x{x.shape[2]}x{x.shape[3]}.npy',std)
        np.save(rf'D:\ricemodify\dataset\withcloud\datasplit\xmyyuanmean{x.shape[0]}x{x.shape[1]}x{x.shape[2]}x{x.shape[3]}.npy',mean)

        # xtrain=(x-mean)/std
        # xtrain[np.isnan(xtrain)]=0
        xtrain=xtrain.reshape(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],self.time_size,self.bandswithindex)
        xtrain=np.transpose(xtrain,(0,3,1,2,4))
        print(xtrain.shape)
       
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
        return x

sample_path = r"D:\ricemodify\dataset\withcloud\riceyuan1andricemy0_10915true_yuechiquxianlinshuisamplewithgridid.csv"
img_yuechi_path =r"D:\ricemodify\dataset\withcloud\yuechi"
img_quxian_path =r'D:\ricemodify\dataset\withcloud\quxian'
img_linshui_path =r'D:\ricemodify\dataset\withcloud\linshui'

patch_size = 11
data_loader = PatchDataLoader(sample_path, img_yuechi_path, img_quxian_path, img_linshui_path, patch_size,time_size=2,band_size=15)
x,y=data_loader.generate_patches()
# patch_df = data_loader.get_data_frame()
print('xy',x.shape,y.shape)
xpath=rf'D:\ricemodify\dataset\withcloud\datasplit\xyuanme_nomeanstd{x.shape[0]}x{x.shape[1]}x{x.shape[2]}x{x.shape[3]}x{x.shape[4]}.npy'
ypath=rf'D:\ricemodify\dataset\withcloud\datasplit\yyuanme_nomeanstd{y.shape[0]}.npy'
# np.save(rf'dataset\withcloud\datasplit\xnomeanstd{x.shape[0]}x{x.shape[1]}x{x.shape[2]}x{x.shape[3]}x{x.shape[4]}.npy',x)
# np.save(rf'dataset\withcloud\datasplit\ynomeanstd{y.shape[0]}.npy',y)