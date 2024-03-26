
import numpy as np
import random
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
class RandomRotation:
    def __init__(self, p=0.5):
        
        self.p = p
    def __call__(self, data_numpy, label_numpy):
        degree =np.random.randint(5,45)
        # if random.random() < self.p:
        rows, cols, _ = data_numpy.shape
        # print(type(data_numpy))
        mat = cv2.getRotationMatrix2D(
            ((cols-1)/2.0, (rows-1)/2.0), degree, 1)
        data_numpy = cv2.warpAffine(data_numpy, mat, (cols, rows))
            # label_numpy = cv2.warpAffine(label_numpy, mat, (cols, rows))
        return data_numpy, label_numpy
        # data_numpy = tf.reshape(data_numpy, (data_numpy.shape[0] , data_numpy.shape[1], 2,16))
        # data_numpy=tf.transpose(data_numpy,(2,0,1,3))


class RandomContrast(object):
    """ Random Contrast """
    
    def __init__(self, contrast=0.4):
        self.contrast = contrast

    def __call__(self, sample,label):
        s = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        mean = np.mean(sample, axis=(0, 1))
        
        return ((sample - mean) * s + mean),label


class RandomChannelDrop(object):
    """ Random Channel Drop """
    def __init__(self, min_n_drop=1, max_n_drop=3):
        self.min_n_drop = min_n_drop
        self.max_n_drop = max_n_drop

    def __call__(self, data_numpy,label):
        n_channels = random.randint(self.min_n_drop, self.max_n_drop)
        channels = np.random.choice(range(data_numpy.shape[-1]), size=n_channels, replace=False)
        # print(channels,'channels')

        for c in channels:
            data_numpy[ :,:,c:c+1] = 0        
        return data_numpy,label 


class RandomBrightness(object):
    """ Random Brightness """
    
    def __init__(self, brightness=0.6,p=0.5):
        self.brightness = brightness
        self.p=p

    def __call__(self, data_numpy,label):
        # if random.random() < self.p:
        s = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        img = data_numpy * s
        
        return img,label

   
class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img, label):
        if random.random() < 0.5:
            return  np.fliplr(img),label
        return img,label
class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    # np.flipud(img) Flip array in the up/down direction.
    def __call__(self, img,label):
        if random.random() < 0.5:
            # print(img.shape)
            flipped_img = np.flipud(img)  # Transpose the array
            print(flipped_img.shape)
            return flipped_img, label
        return img,label


class RandomScale:
    """ Randomly scale the numpy-arrays """

    def __init__(self, scale_range=(0.9, 1.1), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, data_numpy, label_numpy):
        # if random.random() < self.p:
        scale =np.random.random() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        img_h, img_w=11,11
        M_rotate = cv2.getRotationMatrix2D(
            (img_w / 2, img_h / 2), 0, scale)
        # print(M_rotate)
        data_numpy = cv2.warpAffine(data_numpy, M_rotate, (img_w, img_h))
       

        return data_numpy, label_numpy




    
class Compose:
    """ Compose multi augmentation methods"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self,data,label):
    #     for _t in self.transforms:
    #         data, label = _t(data, label)
    #     return data, label
        for transform in self.transforms:
            result = transform(data, label) #this is a common convention in Python to indicate that the variable is intended to be used as a throwaway variable and its value won't be used inside the loop.
            print('result',result)
            if result is not None:
                data, label = result
        return data, label

class RandomApply:
    """ Randomly apply augmentation methods """

    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, data, label):
        for _t in self.transforms:
            if random.random()<self.p:
                data, label = _t(data, label)
        return data, label




            









#     def __getitem__(self, idx):
#         img = self.imgs[idx]
#         field_mask = self.field_masks[idx]
#         if self.split_type == 'train':
#             img, field_mask = self.augment(img, field_mask)
#             img, field_mask = self.crop(img, field_mask)
#         return torch.FloatTensor(img[:, self.feat_arr]), torch.FloatTensor(self.areas[idx:idx+1]), torch.FloatTensor(field_mask), self.gts[idx]
