#%%
import numpy as np
import time
import tensorflow as tf
from keras import backend as K
from keras import callbacks
from model import DualCnn2dGeotimeCbrm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from cfgs import *
from semantic_segmentation.dataset.datagenerate import xtrain,xval,xtest,ytrain,yval,ytest

#%%
def plot(history):
    plt.figure(figsize=(12, 4))
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
    plt.savefig('geo90180twice_cnn2donlyyuechi7973train1994val.jpg')
    # plt.close() 


def accuracy():
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
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--inputshape", type=str)
    # parser.add_argument("--numFilters", type=str)
    # parser.add_argument("--dropout", type=float)
    # parser.add_argument("--learning_rate", type=float)
    # parser.add_argument("--batch_size", type=int)
    # parser.add_argument("--optimizer", type=str)
    # args = parser.parse_args()
    
    model=DualCnn2dGeotimeCbrm(inputshape,channelratio=4,dual=True,dropout=0,L2=0.002) # ratio越小参数越大
    model.compile(optimizer=optimizer ,loss= cross_loss, metrics=metrics)
    print(model.summary())
    model=tf.keras.models.load_model(saveModelPath,custom_objects={"K": K})
    start=time.time()
    history=model.fit(xtrain, ytrain,epochs=30,validation_data=(xval, yval),
    batch_size= batch_size,callbacks=callbacks,verbose=2,shuffle=True)
    a= model.evaluate(xtest, ytest)
    print(a)
    accuracy()
    plot(history)

