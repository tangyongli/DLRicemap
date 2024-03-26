from utils.model.loss import *
from utils.model.modelfunction import *
import tensorflow as tf
from tensorflow import keras
from utils.plot import *
from semantic_segmentation.dataset.transformer import *
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix


def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(999)
   random.seed(999)
   np.random.seed(999)
   tf.random.set_seed(999)
reset_random_seeds()

def dataprogress(savelabelxPath, savelabelyPath,saveunlabelxPath,  savevalxPath, savevalyPath):
    xlabel,ylabel=np.load(savelabelxPath),np.load(savelabelyPath)
    xunlabel=np.load(saveunlabelxPath)
    xval,yval=np.load(savevalxPath),np.load(savevalyPath)
    ylabel= tf.convert_to_tensor(ylabel, dtype=tf.float32)
    ylabel=to_categorical(ylabel,num_classes=2)
    yval= tf.convert_to_tensor(yval, dtype=tf.float32)
    yval=to_categorical(yval, num_classes=2)
    x=np.concatenate([xlabel,xval])
    mean=np.nanmean(x,axis=(0,1,2))
    std=np.nanstd(x,axis=(0,1,2))
    # print('mean',mean.shape,'std',std.shape)
    xlabel=np.where(np.isnan(xlabel), 0, (xlabel - mean) / std)
    xval=np.where(np.isnan(xval), 0, (xval- mean) / std)
    xunlabel=np.where(np.isnan(xunlabel), 0, (xunlabel - mean) / std)
   
    return xlabel,ylabel,xval,yval,xunlabel


def dataagument(xlabel,ylabel,xunlabel,p,strong=1):
    strongcompose_transform = RandomApply([RandomVerticalFlip(),RandomHorizontalFlip(),RandomContrast(),RandomBrightness(),RandomChannelDrop(),RandomRotation(),RandomScale()],p)
    weakcompose_transform = RandomApply([RandomVerticalFlip(),RandomHorizontalFlip(),RandomRotation()],p)
    def strongapply_transforms(x, y):
                if strong:
                    x_transformed, y_transformed = strongcompose_transform(x, y)
                else:
                    x_transformed, y_transformed = weakcompose_transform(x, y)
               
                x_transformed=tf.reshape(x_transformed, (11,11,13))
                
                return x_transformed, y_transformed
    if strong:
        xlabelarray=np.zeros((len(xlabel),11,11,13),dtype=np.float32)
        ylabelarray=np.zeros((len(ylabel),2),dtype=np.float32)
        for i, (x,y) in enumerate(zip(xlabel,ylabel)):
                xlabelarray[i],ylabelarray[i]=strongapply_transforms(x,y)
        return xlabelarray,ylabelarray
    else:
        xunlabelarray=np.zeros((len(xunlabel),11,11,13),dtype=np.float32)
        for i, (x,y) in enumerate(zip(xunlabel,xunlabel)):
           
            xunlabelarray[i],_=strongapply_transforms(x,y)
        return xunlabelarray
           
def cycle(iterable):
    # This function creates an infinite iterator
    while True:
        for x in iterable:
            yield x


# 计算混淆矩阵
def confusion_matrix1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    # 计算生产者精度
    producers_accuracy = np.diag(cm) / np.sum(cm, axis=1)

    # 计算用户精度
    users_accuracy = np.diag(cm) / np.sum(cm, axis=0)
    return producers_accuracy, users_accuracy

def weightsslf(epoch,a,warmupepoch=20):
    if epoch<=warmupepoch:
        weight=0
    elif epoch<30:
        weight=a*(epoch-warmupepoch)/(30-warmupepoch)
    else:
        weight=a
    return weight

def trainsmid(train_dataset,epoch,threshold,weightssl,warmupepoch):
    total_loss = 0
    total_correct=0
    batch_step=0
    total_samples=0
    loss_pseudo=0
    predictionlist=[]
    ylist=[]
    supervisedepochslosslist=[]
    semiviselosslist=[]
    
    # for step in range(steps_per_epoch):
    #     # print(print(next(train_dataset_cycle)))
    #     (x_batch_train, y_batch_train) = next(train_dataset_cycle)
    #     # x_batch_uned = next(weakun_dataset_cycle)
    for x,y in train_dataset:#zip(xlabelarray, ylabel):
        x_batch_train=x
        y_batch_train=y
        x_batch_uned = next(weakun_dataset_cycle)
        # print(x_batch_uned.shape) # numpy
        # print(len(x_batch_uned))
    
        #skip last batch,otherwise it will be higher weight in last batch
        if x_batch_train.shape[0]!=64:
            continue
        if len(x_batch_uned)!=64:
            continue
        total_samples+=x_batch_train.shape[0]
        # print('total_samples',total_samples)
        with tf.GradientTape() as tape:
           
          
            logits_ed = model(x_batch_train,training=True)
            loss_ed = tf.keras.losses.categorical_crossentropy(y_batch_train, logits_ed,from_logits=False) # 32个样本的损失
            supervisedepochslosslist.append(loss_ed)
            # loss_ed=binary_weighted_cross_entropy(y_batch_train,logits_ed,1)
            # loss_ed=binary_balanced_cross_entropy(y_batch_train,logits_ed, smooth=0.0001, beta=0.5)
            # print(loss_ed)
            loss_ed=tf.reduce_mean(loss_ed)
        
            # 计算预测正确的样本数量
            
            predictions = tf.argmax(logits_ed, axis=-1)
            predictionlist.append(predictions)
            ylist.append(y_batch_train)
            correct = tf.reduce_sum(tf.cast(tf.equal(predictions, tf.argmax(y_batch_train, axis=-1)), dtype=tf.float32))
            # print('correct',correct)
            total_correct += correct

            if epoch<=warmupepoch:
               loss_pseudo=0

            if epoch>warmupepoch:
                #预测的概率
                logits_uned = model(x_batch_uned, training=False)
                # 得到具体的预测值0和1
                pseudo_s = tf.argmax(logits_uned, axis=1)
                value,c=np.unique(pseudo_s,return_counts=True)
                # print('predictlabel>0.5',c)
                confidences = tf.reduce_max(logits_uned, axis=-1)
             
                # 返回一个布尔数组，如果满足条件，则为true；否则,false
                mask = confidences >= threshold
                # 返回的是数组，mask为true的伪标签
                pseudo_s_confident = tf.boolean_mask(pseudo_s, mask)
           
                pseudo_s_confident= tf.convert_to_tensor(pseudo_s_confident, dtype=tf.int64)
                pseudo_s_confident=to_categorical(pseudo_s_confident, num_classes=2)
             
                ## 返回的是数组，mask为true的伪标签对应的影像值
                strong_aug_uned = tf.boolean_mask(x_batch_uned, mask).numpy() #tensors
                # print('strong_aug_uned',type(strong_aug_uned))
                # 对tensor进行强增强，由于数据增强的输入是numpy，所以需要先将tensor转变为naary
                if tf.size(strong_aug_uned) > 0:
                    xpresudolabelarray=np.zeros((len(strong_aug_uned),11,11,13),dtype=np.float32)
                    for i, (x,y) in enumerate(zip(strong_aug_uned,strong_aug_uned)):
                        x1,y1=dataagumentarray(1,x,y)
                        xpresudolabelarray[i]=x1
                    # 创建tf.data.dataset, 将numpy转变为tensor，因为model(strongpresudo_dataset, training=True)要求输入的是tensor
                    strongpresudo_dataset= tf.data.Dataset.from_tensor_slices((xpresudolabelarray))
                    # print(' strongpresudo_dataset', strongpresudo_dataset)
                    # strongpresudo_dataset=dataagument( strong_aug_uned,1,42)
                    
                    tensor_list=[]
                    for tensor in strongpresudo_dataset:
                        tensor=tf.reshape(tensor,(-1,11,11,13))
                      
                        tensor_list.append(tensor)
                    
                    strongpresudo_dataset = tf.concat(tensor_list, axis=0)
                    logits_pseudo = model(strongpresudo_dataset, training=True)
                  
                    confidences = tf.reduce_max(logits_pseudo, axis=-1)
                 
                    # 返回一个布尔数组，如果满足条件，则为true；否则,false
                    mask1 = confidences >= threshold
                
                    # 返回的是数组，mask为true的伪标签
                    # pseudo_s_confident = tf.boolean_mask(pseudo_s, mask)
                    # value,count=np.unique(pseudo_s_confident,return_counts=True)
                    # print('predictlabel>0.85',count)
                    #大于阈值的伪标签,0和1
                    # print(' pseudo_s_confident', pseudo_s_confident)
                    # pseudo_s_confident= tf.convert_to_tensor(pseudo_s_confident, dtype=tf.int64)
                    # pseudo_s_confident=to_categorical(pseudo_s_confident, num_classes=2)
                    # pseudoimage = tf.boolean_mask(strongpresudo_dataset, mask1).numpy()
                    
                    # presduosample.add(sample)
                    # print('logits_pseudo',logits_pseudo)
                    loss_pseudo = tf.keras.losses.categorical_crossentropy(pseudo_s_confident, logits_pseudo,from_logits=False)
                    semiviselosslist.append(loss_pseudo)
                    # loss_pseudo=binary_weighted_cross_entropy(pseudo_s_confident,logits_pseudo,1.5)
                    loss_pseudo=tf.reduce_mean(loss_pseudo)
                    # print('loss_pseudotrue',loss_pseudo)
                else:
                    loss_pseudo =0
                    print('loss_pseudo',loss_pseudo)
        
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
      
    avg_loss=total_loss/batch_step
    accuracy = total_correct / total_samples
    ylist=np.concatenate(ylist)
    predictionlist=np.concatenate(predictionlist)
    # print(ylist.shape,predictionlist.shape)
    p,u=confusion_matrix1(tf.argmax(ylist, axis=-1),predictionlist)
    # print('confusion_matrix',p,u)
      
    # print(f"Epoch {epoch+1}, Loss: {total_loss.numpy()/batch_step}")
    return model,avg_loss,accuracy 

def val(model,val_dataset):

    total_correct = 0
    total_samples = 0
    # val_loss = 0
    batch_step=0

    # 遍历验证数据集
    # for (x_batch_val, y_batch_val) in val_dataset:
        # 使用模型进行预测
    y_batch_val=yval
    x_batch_val=xval
    logits_val = model.predict(xval)#model(x_batch_val, training=False)
    # valloss =binary_weighted_cross_entropy(y_batch_val, logits_val,1)#
    valloss=tf.keras.losses.categorical_crossentropy(y_batch_val, logits_val,from_logits=False)
    # print(valloss)
    val_loss=tf.reduce_mean(valloss)
    # 计算预测正确的样本数量
    predictions = tf.argmax(logits_val, axis=-1)
    correct = tf.reduce_sum(tf.cast(tf.equal(predictions, tf.argmax(y_batch_val, axis=-1)), dtype=tf.float32))
    total_correct += correct
    total_samples += x_batch_val.shape[0]
    batch_step+=1
    # val_loss+=valloss
    # 计算预测的准确性
    # print('batch_step',batch_step)
    accuracy = total_correct / total_samples
    # val_loss=val_loss/batch_step
    # losslist.append()
    # print(f"epoch{epoch}Validation Accuracy: {accuracy.numpy()}")
    return accuracy,val_loss
def traintest(savejpgpath,threshold,weightssl,warmupepoch):
    
    best_val_loss= float('inf')
    for epoch in range(1,40):
        train_dataset=tf.data.Dataset.from_tensor_slices((xlabelarray,ylabelarray)).batch(64)
        train_dataset=train_dataset.shuffle(xlabelarray.shape[0])
        model,avg_loss,accuracy=trainsmid(train_dataset,epoch,threshold=threshold,weightssl=weightssl,warmupepoch=warmupepoch)
        # print(accuracy)
        # 监控学习率
       
        # print("Epoch:", epoch, "Learning rate:", optimizer.learning_rate(globalstep).numpy())
        valaccuracy,valloss=val(model,val_dataset=0)
        if valloss < best_val_loss:
            best_val_loss = valloss
         
            # savepath=os.path.join(savemodelpath,f'{saveVersion}.h5')
            model.save(savemodelpath,include_optimizer=True)
        metricdf.at[epoch, "trainloss"] = avg_loss.numpy()
        metricdf.at[epoch, "trainaccuracy"] = accuracy.numpy()
        metricdf.at[epoch, "valaccuracy"] = valaccuracy.numpy()
        metricdf.at[epoch, "valloss"] = valloss.numpy()
        metricdf.to_csv(os.path.join(savemodeldir,f'{saveVersion}.csv'),index=False)
   
    
    plt.figure(figsize=(10,8))
    # Plot training & validation loss values
    plt.subplot(1, 2, 1)

    plt.plot( metricdf["trainloss"])
    plt.plot( metricdf["valloss"])
    plt.title('val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(metricdf["trainaccuracy"])
    plt.plot(metricdf["valaccuracy"])
    plt.title('accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.subplots_adjust(hspace=0.6)
    plt.suptitle(f"{saveVersion}")
    plt.savefig(savejpgpath)
    plt.show()
    time.sleep(5)
    a=3
    bool(print(a))



if __name__ == '__main__':
    # data
 
    savelabelxPath=r"D:\ricemodify\limiteddataset\2022pathsize11\labelsamplesx457.npy"
    savelabelyPath=r"D:\ricemodify\limiteddataset\2022pathsize11\labelsamplesy457.npy"
    savevalxPath=r"D:\ricemodify\limiteddataset\2022pathsize11\samplesvalx217.npy"
    savevalyPath=r"D:\ricemodify\limiteddataset\2022pathsize11\samplesvaly217.npy"
    saveunlabelxPath=r"D:\ricemodify\limiteddataset\2022pathsize11\samplesaddunlabel6483.npy"
   
    
    xlabel,ylabel,xval,yval,xunlabel=dataprogress(savelabelxPath, savelabelyPath,saveunlabelxPath,  savevalxPath, savevalyPath)
    xlabelarray,ylabelarray=dataagument(xlabel,ylabel,xunlabel,p=1,strong=1)
    xunlabelarray=dataagument(xlabel,ylabel,xunlabel,p=1,strong=0)
    weakunlabel_dataset=tf.data.Dataset.from_tensor_slices((xunlabelarray)).batch(64)
    weakun_dataset_cycle = cycle(weakunlabel_dataset)
   

    tf.keras.utils.set_random_seed(999)
    tf.config.experimental.enable_op_determinism()


    # save log setting
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    current_date = current_date .replace(' ', '-').replace(':', '-')
    savemodeldir=r"D:\ricemodify\runRFDL\train\202212median3max_dualseparablecnn2d\log"
    saveVersion=f'{current_date}-crossbatch64_skiplast_datashuffle_lr0.0001x0.95_lossavergebatch_bncnn2d_fixmatch20_30'
    savemodeldir=os.path.join(savemodeldir,f'{saveVersion}')
    os.makedirs(savemodeldir,exist_ok=True)
    savemodelpath=os.path.join(savemodeldir,f'{saveVersion}.h5')
    savejpgpath=os.path.join(savemodeldir,f'1.jpg')
    metricdf=pd.DataFrame(columns=["index","trainloss", "trainaccuracy",'valaccuracy','valloss'])

    # ssl
    threshold=0.95
    warmupepoch=20
    weightssl=1
    # optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0001,
        decay_steps=10*7,
        decay_rate=0.95,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    weight=1
    h,w,c=11,11,13
    patchsize=11
    model=dualsparableCnn2d(inputtag=0,inputshape=(h,w,c),numfilters=3,sattention111=1,sattention011=0,multscalesattetion=0,multscalesattetion001=0,csattention=0,noattention=0,concatdense=0,concatcnntrue1d=1,dropout=dropout,L2=0)
    model.compile(optimizer=optimizer,loss=categorical_crossentropy, metrics=['accuracy'])
    traintest(savejpgpath,threshold,weightssl=weightssl,warmupepoch=warmupepoch)






