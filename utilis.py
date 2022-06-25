# -*- coding: utf-8 -*-
"""
Created on Mar 3 18:14:14 2022

@author: LZK
"""
import numpy as np
from keras import backend as K
import tensorflow as tf


# -----------------------------------------------------------------------#
# 训练集加载
# -----------------------------------------------------------------------#
def load_train_data(train_data):
    print('loading train data...')
    data = np.load(train_data)
    print('Size of train data: ({}, {}, {})'.format(data.shape[0], data.shape[1], data.shape[2]))

    return data


# -----------------------------------------------------------------------#
# 图像预处理
# -----------------------------------------------------------------------#
def Degrade(image, flip):
    sigma = 5 + 20 * np.random.rand(1)
    beta = 0.05 + 0.1 * np.random.rand(1)
    image.astype('float32')
    O_noise = np.random.normal(0, sigma / 255.0, image.shape)  # noise

    if flip:
        G_col = np.random.normal(1, beta, image.shape[1])
        G_noise = np.tile(G_col, (image.shape[0], 1))
        # G_noise = G_noise + np.random.normal(0, 2/255.0, (image.shape[0],image.shape[1]))
        G_noise = np.reshape(G_noise, image.shape)

    else:
        G_col = np.random.normal(1, beta, (image.shape[0], 1))
        G_noise = np.tile(G_col, (1, image.shape[1]))
        # G_noise = G_noise + np.random.normal(0, 2/255.0, (image.shape[0],image.shape[1]))
        G_noise = np.reshape(G_noise, image.shape)

    image_G = np.multiply(image, G_noise)
    image_GO = image_G + O_noise  # input image = clean image + noise
    return image_GO, sigma[0], beta[0]


# -----------------------------------------------------------------------#
# 计算模式
# -----------------------------------------------------------------------#
def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
