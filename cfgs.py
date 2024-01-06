import keras
import os
from keras import callbacks

###############data
sample=12474
samplepath=r'D:\ricemodify\dataset\yuechiquxianlinshui12000samplewithgridid_areasamplebalance4000.csv'#'D:\riceyuechi\dataset\yuechi\label\2022yuechisamplescale100_0rice1other2water.csv'#'D:\ricemodify\dataset\yuechiquxianlinshui12000samplewithgridid_areasamplebalance4000.csv'
imgyuechipath=r"D:\ricemodify\dataset\origin_img\yuechi"
imglinshuipath=r'D:\ricemodify\dataset\origin_img\linshui'
imgquxianpath=r'D:\ricemodify\dataset\origin_img\quxian'
savetrainxPath=r'D:\ricemodify\dataset\datasplit\x12474x2x11x11x15.npy'
savetrainyPath=r'D:\ricemodify\dataset\datasplit\y12474.npy'
saveMeanPath=rf"D:\ricemodify\dataset\datasplit\x{sample}x11x11x24mean.npy"#r'12467x11x11mean121.npy'
saveStdPath=rf"D:\ricemodify\dataset\datasplit\x{sample}x11x11x24std.npy"
#########
inputshape=(2,11,11,16)
patchsize=11
batch_size=64
learning_rate=0.0001
optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
# model.compile(optimizer= opt ,loss='mean_squared_error', metrics=['mse'])
################# loss and metrics
cross_loss=keras.losses.CategoricalCrossentropy(label_smoothing=0)
binaryloss=keras.losses.BinaryCrossentropy() #from_logits=True
metrics=['accuracy',
        ]

############## callbacks
modeltag='Cnn2dGeotimeCbrm56564random2'
dual=1
geotime=4
resnetfilters=['64mp','128mp','256one','128cnn1d','cat64128_128256']
resnetfiltersname='x'.join(map(str, resnetfilters))
print('resnetfiltersname',resnetfiltersname)

saveVersion =f'sample{sample}dual{dual}resnetattention{resnetfiltersname}_geodate{geotime}'
plateauPatience=5
workEnv= 'run'
checkpointDir = os.path.join(workEnv, f'train/patchsize{patchsize}/model') #/train/model is wrong
logsDir= os.path.join(workEnv, f'train/patchsize{patchsize}/history')
plotsDir_ = os.path.join(workEnv, f'train/patchsize11/lossandaccuracy')
confusion_matrixDir = os.path.join(workEnv, f'train/patchsize{patchsize}/confusion_matrix')

os.makedirs(checkpointDir, exist_ok=True)
os.makedirs(logsDir, exist_ok=True)
os.makedirs(plotsDir_, exist_ok=True)
os.makedirs(confusion_matrixDir, exist_ok=True)


# resumeModelPath_ = os.path.join(checkpointDir_, saveVersion_ + '.h5')
saveModelPath = os.path.join(checkpointDir, saveVersion+ '.h5')
print('saveModelPath',saveModelPath)
plotPath_ = os.path.join(plotsDir_, saveVersion + '.png')
confusion_matrixpath=os.path.join(confusion_matrixDir, saveVersion+ '.png')

################predict
predictworkEnv_ = 'run/predict/quxian/patchsize11'
predictarray=os.path.join(predictworkEnv_, 'array')
# predictjpg=os.path.join(predictworkEnv_, 'jpg')
predicttif=os.path.join(predictworkEnv_, 'tif')
startrow,startcol,endrow,endcol=2500,2500,3500,3500
orgintif=r'D:\ricemodify\dataset\origin_img\quxian\quxian_2023sentinel2_maxevi601-901composite15bandswithdoylatlon.tif'






callback_ = [
    callbacks.ModelCheckpoint(monitor='val_accuracy', filepath=saveModelPath, mode='max',
                            save_best_only=True, save_weights_only=False, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=plateauPatience,
                                min_delta=5e-3, min_lr=1e-6, verbose=1),
    callbacks.CSVLogger(filename=os.path.join(logsDir, saveVersion + '.csv')),
    # TensorBoard(log_dir=logsDir_,histogram_freq=1)
]
# callbacks = [
#     callbacks.ModelCheckpoint(
#         monitor='val_accuracy',
#         filepath=saveModelPath,
#         mode='max',
#         save_best_only=True,
#         save_weights_only=False,
#         verbose=1
#     )]
# print(checkpointDir)
# print(saveModelPath)
