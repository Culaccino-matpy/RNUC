# -*- coding: utf-8 -*-
"""
Created on Mar 1 18:12:02 2022

@author: LZK
"""
from keras.layers import Input, Conv2D, Activation, Subtract, Multiply, Concatenate
from keras.models import Model

# L2 = regularizers.l2(1e-6)
L2 = None
init = 'he_normal'


# -----------------------------------------------------------------------#
# 合并式特征提取单元
# -----------------------------------------------------------------------#
def Inception(inpt):
    x_0 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=init)(inpt)
    x_0 = Activation('relu')(x_0)
    x_0 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=init)(x_0)
    x_0 = Activation('relu')(x_0)

    x_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(inpt)
    x_1 = Activation('relu')(x_1)
    x_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(x_1)
    x_1 = Activation('relu')(x_1)

    x_2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer=init)(inpt)
    x_2 = Activation('relu')(x_2)
    x_2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer=init)(x_2)
    x_2 = Activation('relu')(x_2)

    x = Concatenate()([x_0, x_1, x_2])

    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=init)(x)
    x = Activation('relu')(x)
    return x


# -----------------------------------------------------------------------#
# 级联两个网络结构
# -----------------------------------------------------------------------#
def NUCNN():
    inpt = Input(shape=(None, None, 1))
    x = Inception(inpt)
    x = Inception(x)
    x = Inception(x)
    x = Inception(x)
    x = Inception(x)
    # x = Inception(x)
    # x = Inception(x)
    O = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=init)(x)
    T = Subtract()([inpt, O])

    x = Inception(T)
    x = Inception(x)
    x = Inception(x)
    x = Inception(x)
    x = Inception(x)
    G = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=init)(x)
    res = Multiply()([T, G])
    model = Model(inputs=inpt, outputs=[res])

    return model
