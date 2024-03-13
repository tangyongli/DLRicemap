from utils.model.loss import *
from utils.model.modelfunction import *
import tensorflow as tf
from tensorflow import keras
from semantic_segmentation.dataset.transformer import *
def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(999)
   random.seed(999)
   np.random.seed(999)
   tf.random.set_seed(999)
reset_random_seeds()
learning_rate=0.0001
optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
weight=1
h,w,c=11,11,13
loss=tf.keras.losses.CategoricalCrossentropy()
model=dualsparableCnn2d(inputtag=0,inputshape=(h,w,c),numfilters=3,sattention111=1,sattention011=0,multscalesattetion=0,multscalesattetion001=0,csattention=0,noattention=0,concatdense=0,concatcnntrue1d=1,dropout=dropout,L2=0)#inputshape,channelratio,bandswithindex=False,bandposition=14,indexposition=17,sar=False,dropout=0.2,L2=0simplecnn2d(inputshape=(h,w,c),num_filters=[32,64,128,256,256],dropratio=0)#inputshape,channelratio,bandswithindex=False,bandposition=14,indexposition=17,sar=False,dropout=0.2,L2=0simplecnn2d(inputshape=(h,w,c),num_filters=[32,64,128,256,256],dropratio=0)

saveVersion=f'samplesaddgeetrain_block10val2classes_depthwiseatte111catcnn1dcbrm1_batchnorcnn2d1ddense'
print(saveVersion)
print('222222222') 



savelabelxPath=r"D:\ricemodify\limiteddataset\2022pathsize11\labelsamplesx457.npy"
savelabelyPath=r"D:\ricemodify\limiteddataset\2022pathsize11\labelsamplesy457.npy"
savevalxPath=r"D:\ricemodify\limiteddataset\2022pathsize11\samplesvalx217.npy"
savevalyPath=r"D:\ricemodify\limiteddataset\2022pathsize11\samplesvaly217.npy"

saveunlabelxPath=r"D:\ricemodify\limiteddataset\2022pathsize11\samplesaddunlabel6483.npy"
patchsize=11

def dataprogress(savelabelxPath, savelabelyPath,saveunlabelxPath,  savevalxPath, savevalyPath):
    xlabel,ylabel=np.load(savelabelxPath),np.load(savelabelyPath)
    xunlabel=np.load(saveunlabelxPath)
    xval,yval=np.load(savevalxPath),np.load(savevalyPath)
    print(np.unique(ylabel,return_counts=True)) #(array([0, 1]), array([244, 213], dtype=int64))
    
    ylabel= tf.convert_to_tensor(ylabel, dtype=tf.float32)
    ylabel=to_categorical(ylabel,num_classes=2)
    yval= tf.convert_to_tensor(yval, dtype=tf.float32)
    yval=to_categorical(yval, num_classes=2)
    x=np.concatenate([xlabel,xval])
    mean=np.nanmean(x,axis=(0,1,2))
    std=np.nanstd(x,axis=(0,1,2))
    print('mean',mean.shape,'std',std.shape)
    xlabel=np.where(np.isnan(xlabel), 0, (xlabel - mean) / std)
    xval=np.where(np.isnan(xval), 0, (xval- mean) / std)
    xunlabel=np.where(np.isnan(xunlabel), 0, (xunlabel - mean) / std)
   
    return xlabel,ylabel,xval,yval,xunlabel
   
xlabel,ylabel,xval,yval,xunlabel=dataprogress(savelabelxPath, savelabelyPath,saveunlabelxPath,  savevalxPath, savevalyPath)

def dataagumentarray(strongagu,x,y):
    if strongagu:
        # x,y=RandomCrop()(x,y)
        # # print(x.shape,y.shape)
        # x,y=CenterCrop()(x,y)
        # print(x.shape,y.shape)
        x,y=RandomVerticalFlip()(x, y) #
        x,y=RandomHorizontalFlip()(x,y)
        x,y=RandomRotation()(x,y)
        # x,y=RandomScale()(x, y)
        # x,y=RandomContrast()(x,y)
        # x,y= RandomChannelDrop()(x,y)
        # x,y=RandomBrightness()(x,y)
    else:
        x,y=RandomVerticalFlip()(x, y) 
        x,y=RandomHorizontalFlip()(x,y)
        x,y=RandomRotation()(x,y)

    return x,y
xlabelarray=np.zeros((len(xlabel),11,11,13),dtype=np.float32)
ylabelarray=np.zeros((len(ylabel),2),dtype=np.float32)
def apply_transforms(strongagu,x, y=None):
        x_transformed, y_transformed = dataagumentarray(strongagu,x, y)
        return x_transformed, y_transformed

for i, (x,y) in enumerate(zip(xlabel,ylabel)):
    x1,y1=dataagumentarray(1,x,y)
    xlabelarray[i],ylabelarray[i]=x1,y1

print(xlabelarray.shape,ylabelarray.shape)

xunlabelarray=np.zeros((len(xunlabel),11,11,13),dtype=np.float32)
for i, (x,y) in enumerate(zip(xunlabel,xunlabel)):
    x1,y1=dataagumentarray(0,x,y)
    xunlabelarray[i]=x1

train_dataset=tf.data.Dataset.from_tensor_slices((xlabelarray,ylabelarray)).batch(32)
weakunlabel_dataset=tf.data.Dataset.from_tensor_slices((xunlabelarray)).batch(2048)
val_dataset=tf.data.Dataset.from_tensor_slices((xval,yval)).batch(32)
def cycle(iterable):
    # This function creates an infinite iterator
    while True:
        for x in iterable:
            yield x

# Create infinite iterators
train_dataset_cycle = cycle(train_dataset) 
weakun_dataset_cycle = cycle(weakunlabel_dataset)



# Hyperparameters
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
batch_size = 32
threshold = 0.9
epochs = 10
r = 1 # Weight for the pseudo- loss

next(weakun_dataset_cycle)
next(weakun_dataset_cycle)
next(weakun_dataset_cycle)
next(weakun_dataset_cycle)
next(weakun_dataset_cycle)
next(weakun_dataset_cycle)
next(weakun_dataset_cycle)
# Get the number of steps per epoch
steps_per_epoch = len(list(train_dataset))#, len(list(weakunlabel_dataset))

def weightsslf(epoch,a):
    if epoch<20:
        weight=0
    elif epoch<40:
        weight=a*(epoch-10)/(20-10)
    else:
        weight=a
    return weight

from sklearn.manifold import TSNE
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# 计算混淆矩阵
def confusion_matrix1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    # 计算生产者精度
    producers_accuracy = np.diag(cm) / np.sum(cm, axis=1)

    # 计算用户精度
    users_accuracy = np.diag(cm) / np.sum(cm, axis=0)
    return producers_accuracy, users_accuracy
def trainsmid(train_dataset_cycle, weakun_dataset_cycle, steps_per_epoch,threshold,weightssl,warmupepoch):
    total_loss = 0
    total_correct=0
    batch_step=0
    total_samples=0
    loss_pseudo=0
    predictionlist=[]
    ylist=[]
    
    for step in range(steps_per_epoch):
        # print(print(next(train_dataset_cycle)))
        (x_batch_train, y_batch_train) = next(train_dataset_cycle)
        x_batch_uned = next(weakun_dataset_cycle)
        # print( next(weakun_dataset_cycle).shape)
        # print( next(weakun_dataset_cycle)[0:1,0:1,0:1,...])
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxx')
        # print( next(weakun_dataset_cycle)[0:1,0:1,0:1,...])
        total_samples+=x_batch_train.shape[0]
        # print(x_batch_train.shape)
        with tf.GradientTape() as tape:
           
            logits_ed = model(x_batch_train,training=True)
            
            # confidences = tf.nn.softmax(logits_ed, axis=-1)
            # print('confidences',confidences)
            # print('logits_ed',logits_ed)
            loss_ed = tf.keras.losses.categorical_crossentropy(y_batch_train, logits_ed,from_logits=False) # 32个样本的损失
            # loss_ed = tf.keras.losses.categorical_crossentropy(y_batch_train, logits_ed,from_logits=True) # 32个样本的损失
            loss_ed=tf.reduce_mean(loss_ed)
            # loss= tf.keras.losses.CategoricalCrossentropy(from_logits=True)，这里的loss是一个训练batch中所有损失的平均值
            # loss_ed = loss(y_batch_train, logits_ed)
            # with open ('./loss_softmax_true.txt', 'a') as f:
            #     f.write(
            #         str(loss_ed.numpy()) + '\n')
            #     f.close
            # print('loss_ed',loss_ed)
            # 计算预测正确的样本数量
            
            predictions = tf.argmax(logits_ed, axis=-1)
            predictionlist.append(predictions)
            ylist.append(y_batch_train)
            correct = tf.reduce_sum(tf.cast(tf.equal(predictions, tf.argmax(y_batch_train, axis=-1)), dtype=tf.float32))
            total_correct += correct
           

  

            if epoch<warmupepoch:
               loss_pseudo=0

            if epoch>=warmupepoch:
                #预测的概率
                logits_uned = model(x_batch_uned, training=False)
                print('logits_uned1',logits_uned)
                # print('logitsunedsum',tf.reduce_sum(logits_uned,axis=-1))
                # logits_uned1 = tf.nn.softmax(logits_uned, axis=1)
                # print('logits_uned2',logits_uned1)
                # 得到具体的预测值0和1
                pseudo_s = tf.argmax(logits_uned, axis=1)
                value,c=np.unique(pseudo_s,return_counts=True)
                print('predictlabel>0.5',c)
                confidences = tf.reduce_max(logits_uned, axis=-1)
             
                # print('confidences',confidences)
                # 返回一个布尔数组，如果满足条件，则为true；否则,false
                mask = confidences >= threshold
            
                # 返回的是数组，mask为true的伪标签
                pseudo_s_confident = tf.boolean_mask(pseudo_s, mask)
                value,count=np.unique(pseudo_s_confident,return_counts=True)
                print('predictlabel>0.85',count)
                #大于阈值的伪标签,0和1
                # print(' pseudo_s_confident', pseudo_s_confident)


                pseudo_s_confident= tf.convert_to_tensor(pseudo_s_confident, dtype=tf.int64)
                pseudo_s_confident=to_categorical(pseudo_s_confident, num_classes=2)
                # print('pseudo_s_confident',pseudo_s_confident.shape)
                ## 返回的是数组，mask为true的伪标签对应的影像值
                strong_aug_uned = tf.boolean_mask(x_batch_uned, mask).numpy() #tensors
                # print('strong_aug_uned',type(strong_aug_uned))
                # 对tensor进行强增强，由于数据增强的输入是numpy，所以需要先将tensor转变为naary

                if tf.size(strong_aug_uned) > 0:
                    xpresudolabelarray=np.zeros((len(strong_aug_uned),11,11,13),dtype=np.float32)
                    for i, (x,y) in enumerate(zip(strong_aug_uned,strong_aug_uned)):
                        # print(x)
                        x1,y1=dataagumentarray(1,x,y)
                        xpresudolabelarray[i]=x1
                    # 创建tf.data.dataset, 将numpy转变为tensor，因为model(strongpresudo_dataset, training=True)要求输入的是tensor
                    strongpresudo_dataset= tf.data.Dataset.from_tensor_slices((xpresudolabelarray))
                    # print(' strongpresudo_dataset', strongpresudo_dataset)
                    # strongpresudo_dataset=dataagument( strong_aug_uned,1,42)
                    
                    tensor_list=[]
                    for tensor in strongpresudo_dataset:
                        # print('tensor', tensor.shape)
                        tensor=tf.reshape(tensor,(-1,11,11,13))
                        # Add the tensor to the list
                        tensor_list.append(tensor)
                    
                    strongpresudo_dataset = tf.concat(tensor_list, axis=0)

                    # print('strongpresudo_dataset',strongpresudo_dataset.shape)
                    # print('strongpresudo_dataset',strongpresudo_dataset[0])
                    # print('strongpresudo_dataset',strongpresudo_dataset[1])
                   

    
                    logits_pseudo = model(strongpresudo_dataset, training=True)
                    print('logits_pseudo',logits_pseudo)
                    # print('logitssum',tf.reduce_sum(logits_pseudo,axis=-1))
                    # presudo_confidences = tf.nn.softmax(logits_pseudo, axis=-1)
                 
                    # print('presudo_confidences',presudo_confidences)
                    loss_pseudo = tf.keras.losses.categorical_crossentropy(pseudo_s_confident, logits_pseudo,from_logits=False)
                    loss_pseudo=tf.reduce_mean(loss_pseudo)
                    # print('loss_pseudotrue',loss_pseudo)
                else:
                    loss_pseudo =0
                    # print('loss_pseudo',loss_pseudo)
        
            # 每一个batch中的混合损失
            r=weightsslf(epoch, weightssl)
            combine_loss = loss_ed + r * loss_pseudo
            # print('supervisedloss',loss_ed)
            # print('combine_loss',combine_loss)
    
        #利用每个batch中的混合损失更新梯度，然后更新模型参数
        grads = tape.gradient(combine_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # 累计一轮中多个batch的损失
        total_loss += combine_loss
        # print(f'batchstep{batch_step},total_loss is:',total_loss)
        batch_step += 1
        # print('batch_stepinside', batch_step)

                
    # print('batch_step', batch_step)
   
    avg_loss=total_loss/batch_step
    accuracy = total_correct / total_samples
    ylist=np.concatenate(ylist)
    predictionlist=np.concatenate(predictionlist)
    print(ylist.shape,predictionlist.shape)
    p,u=confusion_matrix1(tf.argmax(ylist, axis=-1),predictionlist)
    print('confusion_matrix',p,u)
           
    # 虽然最后一个batch的样本数量可能不一样，但是由于之前是在每个batch取损失均值，最后可以将batch损失均值的总和除以batch数量；因为每个样本的权重一样。
    # 也可以不对batch的损失进行平均，直接计算每个batch中的总损失，然后直接将所有batch的总损失除以样本总数。
    print(f"Epoch {epoch+1}, Loss: {total_loss.numpy()/batch_step}")
    with open ('./avg_loss_sofmax_false1.txt', 'a') as f:
                f.write(
                    str(avg_loss.numpy()) + '\n')
                f.close
    model.save('m.h5')
    return model,avg_loss,accuracy 

def val(model,val_dataset):
    total_correct = 0
    total_samples = 0
    val_loss = 0
    batch_step=0

    # 遍历验证数据集
    for (x_batch_val, y_batch_val) in val_dataset:
        # 使用模型进行预测
        logits_val = model(x_batch_val, training=False)
        valloss =tf.keras.losses.categorical_crossentropy(y_batch_val, logits_val,from_logits=False)
        valloss=tf.reduce_mean(valloss)
        # 计算预测正确的样本数量
        predictions = tf.argmax(logits_val, axis=-1)
        correct = tf.reduce_sum(tf.cast(tf.equal(predictions, tf.argmax(y_batch_val, axis=-1)), dtype=tf.float32))
        total_correct += correct
        total_samples += x_batch_val.shape[0]
        batch_step+=1
        val_loss+=valloss
    # 计算预测的准确性
    # print('batch_step',batch_step)
    accuracy = total_correct / total_samples
    val_loss=val_loss/batch_step
    # losslist.append()
    # print(f"Validation Accuracy: {accuracy.numpy()}")
    return accuracy,val_loss

import pandas as pd
metricdf=pd.DataFrame(columns=["index","trainloss", "trainaccuracy",'valaccuracy','valloss'])
for epoch in range(20):
    tf.keras.utils.set_random_seed(999)
    tf.config.experimental.enable_op_determinism()
    model,avg_loss,accuracy=trainsmid(train_dataset_cycle, weakun_dataset_cycle, steps_per_epoch,0.85,1,5)
    valaccuracy,valloss=val(model,val_dataset)
    metricdf.at[epoch, "trainloss"] = avg_loss
    metricdf.at[epoch, "trainaccuracy"] = accuracy
    metricdf.at[epoch, "valaccuracy"] = valaccuracy
    metricdf.at[epoch, "valloss"] = valloss
    

# 绘制损失曲线
plt.figure(figsize=(8, 10))
# Plot training & validation loss values
plt.subplot(2, 2, 1)

plt.plot( metricdf["trainloss"])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 2, 2)
plt.plot(metricdf["trainaccuracy"])
plt.title('Training accuracy')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.subplots_adjust(hspace=0.6)

plt.subplot(2, 2, 3)
plt.plot( metricdf["valloss"])
plt.title('val Loss')
plt.xlabel('Epoch')
plt.ylabel('val Loss')

plt.subplot(2, 2, 4)
plt.plot(metricdf["valaccuracy"])
plt.title('val accuracy')
plt.xlabel('Epoch')
plt.ylabel('val accuracy')

# Adjust the space between subplots
  # Adjust the vertical spacing

# Alternatively, you can use plt.tight_layout() to automatically adjust the parameters
# plt.tight_layout()
plt.suptitle("train and valiation curve,lr0.0001,batch32")
# plt.savefig(f'./trainvalcurve_softmax_crossformlogittrue.png')
plt.show()


