import numpy as np
import time

# from model.models import DualCnn2dGeotimeCbrm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import to_categorical

from RF_DL.plot import *


def plot(history,saveVersion,plotPath):
    plt.figure(figsize=(16, 4))
    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.suptitle(saveVersion)
    plt.tight_layout()
    # plt.show() # 渲染了图像可以保存，但是保存的是空白，正确的顺序是不要渲染
    plt.savefig(plotPath) #plt.savefig(...) will then create an additional figure which might be why you end up with an open figure in the end.
    plt.show()
    time.sleep(5)
    # # Clear the current axes.
    # plt.cla() 
    # # Clear the current figure.
    # plt.clf() 
    # # Closes all the figure windows.
    # plt.close('all')   
    # # plt.close(fig)
    # # gc.collect()
    # # plt.close() 


def accuracy():
    loss,accuracy= model.evaluate(xtest, ytest)
    pred=model.predict(xtest)
    maxpred=np.argmax(pred,axis=-1)
    maxpred.shape
    ytest1=np.argmax(ytest,axis=-1)
    conf_mat = confusion_matrix(ytest1, maxpred)
    # Plot confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(maxpred), yticklabels=np.unique(maxpred))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(saveVersion)
    plt.text(conf_mat.shape[1] + 0.4, conf_mat.shape[0] - 1.2, f'Loss: {loss:.2f}', ha='left', va='center', color='red')
    plt.text(conf_mat.shape[1] + 0.4, conf_mat.shape[0] - 1, f'Accuracy: {accuracy:.2f}', ha='left', va='center', color='red')
    # plt.text(enddoy, evi_value, f'({enddoy}, {end_evi_value:.2f})', ha='left', va='top')
    # plt.show()
    plt.legend(loc='upper left')
    plt.savefig(confusion_matrixpath)
    # plt.show()
    # time.sleep(5)
    print('plt',plt)
    # plt.cla() 
    # # Clear the current figure.
    # plt.clf() 
    # # Closes all the figure windows.
    # plt.close('all')   

def loadandprocessdata(x,y,fillvalue=0,patchsize=11,includeindex=True):
    x=np.load(x)
    y=np.load(y)
    print(x.shape)
    patchshape,channels=x.shape[2],x.shape[-1]
    print(patchshape,channels)
    x=x[...,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,0:channels] # sample,2,11,11,15
    print('x',x.shape)
    print('y',y.shape)
    y=np.where(y!=0,1,y)
    if len(y.shape)==1:
        y= tf.convert_to_tensor(y, dtype=tf.float32)
        y=to_categorical(y, num_classes=2)
    mean=np.nanmean(x,axis=(0,1,2))
    std=np.nanstd(x,axis=(0,1,2))
    x= np.where(np.isnan(x), 0, (x - mean) / std)
    # xcenter=x[...,5:6,5:6,0:39]
    # mask=~np.isnan(x) #(10658, 11, 11, 39)
    # print(mask.shape)
    # x=x[mask]
    # print(x.shape)
    # y=y[mask]
    # nan=np.isnan(x).any(axis=(1,2,3))
    # mask=~nan
    # print(mask.shape)
    # x=x[mask]
    # y=y[mask]
    x[np.isnan(x)]=fillvalue
    ratio = np.sum(x== fillvalue) / x.size #6376*49*121
    print(f"值为0的比率: {ratio:.2%}") #值为-1的比率: 1.61%'
    return x,y