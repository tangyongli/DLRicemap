import keras
import os
from keras import callbacks

###############data
sample=10915#8319#6114
samplepath=r"D:\ricemodify\dataset\withcloud\riceyuan1andricemy0_10915true_yuechiquxianlinshuisamplewithgridid.csv"#"D:\ricemodify\dataset\riceyuanandricemy_yuechiquxianlinshui3671_4648samplewithgridid.csv"#'D:\ricemodify\dataset\yuechiquxianlinshui12000samplewithgridid_areasamplebalance4000.csv'#'D:\riceyuechi\dataset\yuechi\label\2022yuechisamplescale100_0rice1other2water.csv'#'D:\ricemodify\dataset\yuechiquxianlinshui12000samplewithgridid_areasamplebalance4000.csv'
imgyuechipath=r"D:\ricemodify\dataset\withcloud\yuechi"
imglinshuipath=r'D:\ricemodify\dataset\withcloud\linshui'
imgquxianpath=r'D:\ricemodify\dataset\withcloud\quxian'
savetrainxPath=rf'D:\ricemodify\dataset\s1s2cloud70mask\xmyyuan{sample}x2x11x11x17.npy'#'D:\ricemodify\dataset\datasplit\x12474x2x11x11x15.npy'#'D:\ricemodify\dataset\datasplit\x12474x2x33x33x15meanstd.npy'
savetrainyPath=rf'D:\ricemodify\dataset\s1s2cloud70mask\ymyyuan{sample}x2x11x11x17.npy'#D:\ricemodify\dataset\datasplit\y12474.npy'#'D:\ricemodify\dataset\datasplit\y12473meanstd.npy'#r'D:\ricemodify\dataset\datasplit\y12474meanstd.npy'
saveMeanPath=rf"D:\ricemodify\dataset\s1s2cloud70mask\xmyyuanmean{sample}x11x11x28.npy"#D:\ricemodify\dataset\datasplit\x12474x11x11x24mean.npy'#rf"D:\ricemodify\dataset\datasplit\xmean{sample}x33x33x24.npy"#
saveStdPath=rf"D:\ricemodify\dataset\s1s2cloud70mask\xmyyuanstd{sample}x11x11x28.npy"#D:\ricemodify\dataset\datasplit\x12474x11x11x24std.npy'#rf"D:\ricemodify\dataset\datasplit\xstd{sample}x33x33x24.npy"
#########
patchsize=11
inputchannel=9
inputshape=(patchsize,patchsize,inputchannel)
times,inputheight,inputwidth=1,11,11
geotimebands=4
bandswithindex,bandswithoutindex=15,12#12,9
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
resnetfilters=['64mp','128mp','128two','128cnn1d','cat64128_128128']
resnetfiltersname='x'.join(map(str, resnetfilters))
print('resnetfiltersname',resnetfiltersname)
saveVersion=f's1vhmaxminratiosample{sample}input{patchsize}x{patchsize}x{inputchannel}_imggeneraterotatflip_dual{dual}dwise_lastattention' #corrdattentionhswish
# saveVersion =f'sample{sample}imggeneraterotatflip_dual{dual}resnetattentionbeforecnn1d{resnetfiltersname}_geodate{geotime}'
# saveVersion =f'sample{sample}classrotatflip_dual{dual}resnetattentionafter{resnetfiltersname}_geodate{geotime}'
plateauPatience=5
workEnv= 'D:/ricemodify/run/'
checkpointDir = os.path.join(workEnv, f'train/patchsize{patchsize}/model') #/train/model is wrong
logsDir= os.path.join(workEnv, f'train/patchsize{patchsize}/history')
plotsDir_ = os.path.join(workEnv, f'train/patchsize{patchsize}/lossandaccuracy')
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
predictworkEnv_ = 'run/predict/'

startrow,startcol,endrow,endcol=2000,2000,3000,3000#1500,1500,4000,4000







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
