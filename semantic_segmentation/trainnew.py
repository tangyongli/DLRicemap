#%%
import numpy as np
import time
import random
import tensorflow as tf
from keras import backend as K
from keras import callbacks
# from model.models import DualCnn2dGeotimeCbrm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from cfgs import *
from semantic_segmentation.dataset.transformer_ImageDataGenerator import dataagument,val_data_generator#augment_data
from semantic_segmentation.dataset import transformer
from semantic_segmentation.dataset.transformer import  Compose,RandomRotation,RandomContrast,RandomScale,RandomHorizontalFlip,RandomVerticalFlip
from keras.utils import to_categorical
from semantic_segmentation.dataset.loaddataset import DataLoader
# from semantic_segmentation.dataset.datagenerate import xtrain,xval,xtest,ytrain,yval,ytest
from semantic_segmentation.model.modelclass import DataAndModelLoader
def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(999)
   random.seed(999)
   np.random.seed(999)
   tf.random.set_seed(999)
reset_random_seeds()
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
    plt.savefig(plotPath_) #plt.savefig(...) will then create an additional figure which might be why you end up with an open figure in the end.
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
    plt.show()
    time.sleep(5)
    print('plt',plt)
    # plt.cla() 
    # # Clear the current figure.
    # plt.clf() 
    # # Closes all the figure windows.
    # plt.close('all')   


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--inputshape", type=str)
    # parser.add_argument("--numFilters", type=str)
    # parser.add_argument("--dropout", type=float)
    # parser.add_argument("--learning_rate", type=float)
    # parser.add_argument("--batch_size", type=int)
    # parser.add_argument("--optimizer", type=str)
    # args = parser.parse_args()
   
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    savetrainxPath =r'D:\ricemodify\dataset\s1s2medianmaxmincloudmask\xmyyuan10915x1x11x11x19.npy'#D:\ricemodify\dataset\withcloud\datasplit\xnomeanstd10915x2x11x11x15.npy',r'D:\ricemodify\dataset\withcloud\datasplit\ynomeanstd10915.npy'#D:\ricemodify\dataset\datasplit\xnomeanstd12473x2x33x33x15.npy',r'D:\ricemodify\dataset\datasplit\ynomeanstd12473.npy'
    savetrainyPath = r'D:\ricemodify\dataset\s1s2medianmaxmincloudmask\ymyyuan10915x1x11x11x19.npy'
    data_and_model_loader = DataAndModelLoader(x=savetrainxPath, y=savetrainyPath,patch_size=11,include_index=False,bandposition=14,indexposition=17)
    # 调用方法
    history=data_and_model_loader.load_data_and_train_model()
    model,xtest,ytest=tf.keras.models.load_model(saveModelPath)
    if xtest.shape[1]==1:
        xtest=np.squeeze(xtest,axis=1)
    a= model.evaluate(xtest, ytest)
    print('accuracy',a)
    accuracy()

    plot(history)
    

    # history = model.fit_generator(mytrain, steps_per_epoch=steps_per_epoch,validation_data=(xval,yval),
    #             epochs=10,callbacks=callback_,verbose=2,shuffle=True)
    # # history=model.fit(train_dataset,epochs=15,validation_data=(xval, yval),
    # # batch_size= batch_size,callbacks=callback_,verbose=2,shuffle=True)
    # model=tf.keras.models.load_model(saveModelPath,custom_objects={"K": K})
    # a= model.evaluate(xtest, ytest)
    # print(a)
    # accuracy()
    # plot(history)