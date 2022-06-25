# -*- coding: utf-8 -*-
"""
Created on Mar 6 18:47:14 2022

@author: LZK
"""
from keras import backend as K
import tensorflow as tf
import os, time, glob
import PIL.Image as Image
import numpy as np
import pandas as pd
import keras
import cv2
import time
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from keras.optimizers import Adam, SGD
from skimage.measure import compare_psnr, compare_ssim
import models
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract, Multiply, Add, Concatenate
from keras import regularizers
from keras.utils import plot_model
from keras import initializers
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from models import NUCNN
from utilis import Degrade, PSNR, load_train_data


# from util import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# # Params


# -----------------------------------------------------------------------#
# 训练数据生成data generation
# -----------------------------------------------------------------------#
def train_datagen(y_, batch_size=128):
    # y_ is the tensor of clean patches
    indices = list(range(y_.shape[0]))
    steps_per_epoch = len(indices) // batch_size - 1
    j = 0
    while (True):
        np.random.shuffle(indices)  # shuffle
        ge_batch_y = []
        ge_batch_x = []
        for i in range(batch_size):
            flip = True  # (i%2 == 0)
            sample = y_[indices[j * batch_size + i]]
            sample_GO, _, _ = Degrade(sample, flip)  # input image = clean image + noise
            ge_batch_y.append(sample)
            ge_batch_x.append(sample_GO)
        if j == steps_per_epoch:
            j = 0
            np.random.shuffle(indices)
        else:
            j += 1
        yield np.array(ge_batch_x), np.array(ge_batch_y)


# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#

def adam_step_decay(epoch):
    if epoch <= 25:
        lr = 1e-3
    elif epoch > 25 and epoch <= 45:
        lr = 1e-4
    else:
        lr = 1e-5

    return lr


# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#

batch_size = 128
train_data = './data/Train/clean_patches.npy'
epoch = 50
save_every = 1
pretrain = None
# pretrain = './checkpoints/1_plain_adam/weights-04-27.7000-0.0017.hdf5'
init_epoch = 0

multi_GPU = True

TRAIN = False
TEST = not TRAIN
realFrame = True
# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#

if TRAIN:
    data = load_train_data(train_data)
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))
    data = data.astype('float32') / 255.0
    if multi_GPU:
        with tf.device('/cpu:0'):
            model = NUCNN()
    else:
        model = NUCNN()
    # model selection
    with open('./export/model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    plot_model(model, to_file='./export/model.png')
    opt = Adam(decay=1e-6)
    # opt = SGD(momentum=0.9, decay=1e-4, nesterov=True)
    if multi_GPU:
        print('Using Multi GPUs !')
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=opt, loss='mse', metrics=[PSNR])
    else:
        model.compile(optimizer=opt, loss='mse', metrics=[PSNR])

    if pretrain:
        print('Load pretrained model !')
        if multi_GPU:
            parallel_model.load_weights(pretrain)
        else:
            model.load_weights(pretrain)
    # use call back functions
    filepath = "./checkpoints/weights-{epoch:02d}-{PSNR:.4f}-{loss:.4f}.hdf5"
    ckpt = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, period=save_every, save_weights_only=True)
    lr = LearningRateScheduler(adam_step_decay)
    TensorBoard = keras.callbacks.TensorBoard(log_dir='./logs')
    # train 
    if multi_GPU:
        history = parallel_model.fit_generator(train_datagen(data, batch_size=batch_size),
                                               steps_per_epoch=len(data) // batch_size,
                                               epochs=epoch, verbose=1, callbacks=[ckpt, lr, TensorBoard],
                                               initial_epoch=init_epoch)
    else:
        history = model.fit_generator(train_datagen(data, batch_size=batch_size),
                                      steps_per_epoch=len(data) // batch_size,
                                      epochs=epoch, verbose=1, callbacks=[ckpt, lr, TensorBoard],
                                      initial_epoch=init_epoch)

# ----------------------------------------------------------------------#
# ----------------------------------------------------------------------#
# ----------------------------------------------------------------------#
# ----------------------------------------------------------------------#
if TEST:
    if not realFrame:
        save_dir = 'results/sim'
        test_dir = 'data/Test/Set12'

        WEIGHT_PATH = './checkpoints/weights-50-28.8004-0.0013.hdf5'


        # ----------------------------------------------------------------------#
        def Addnoise(image, sigma=11.55, beta=0.15):
            image.astype('float32')
            O_noise = np.random.normal(0, sigma / 255.0, image.shape)  # noise

            G_col = np.random.normal(1, beta, image.shape[1])
            G_noise = np.tile(G_col, (image.shape[0], 1))
            # G_noise = G_noise + np.random.normal(0, 2/255.0, (image.shape[0],image.shape[1]))
            G_noise = np.reshape(G_noise, image.shape)

            image_G = np.multiply(image, G_noise)
            image_GO = image_G + O_noise  # input image = clean image + noise
            return image_GO


        # ----------------------------------------------------------------------#

        if multi_GPU:
            model = NUCNN()
            print('Using Multi GPUs !')
            model = multi_gpu_model(model, gpus=2)
            model.load_weights(WEIGHT_PATH)
        else:
            model = NUCNN()
            model.load_weights(WEIGHT_PATH)
        print('Start to test on {}'.format(test_dir))
        out_dir = save_dir + '/' + test_dir.split('/')[-1] + '/'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        name = []
        GainList = []
        OnoiseList = []
        psnr = []
        ssim = []
        file_list = glob.glob('{}/*.*'.format(test_dir))
        for file in file_list:
            start = time.time()
            if file[-3:] in ['bmp', 'jpg', 'png', 'BMP']:
                # read image
                img_clean = np.array(Image.open(file), dtype='float32') / 255.0
                img_test = Addnoise(img_clean, sigma=11.55, beta=0.15).astype('float32')
                # predict
                x_test = img_test.reshape(1, img_test.shape[0], img_test.shape[1], 1)
                y_predict = model.predict(x_test)
                # calculate numeric metrics
                img_out = y_predict.reshape(img_clean.shape)
                img_out = np.clip(img_out, 0, 1)
                psnr_noise, psnr_denoised = compare_psnr(img_clean, img_test), compare_psnr(img_clean, img_out)
                ssim_noise, ssim_denoised = compare_ssim(img_clean, img_test), compare_ssim(img_clean, img_out)
                psnr.append(psnr_denoised)
                ssim.append(ssim_denoised)
                # save images
                filename = file.split('/')[-1].split('.')[0]  # get the name of image file
                filename = filename[6:]
                name.append(filename)
                img_test = Image.fromarray(np.clip((img_test * 255), 0, 255).astype('uint8'))
                img_test.save(out_dir + filename + '_psnr{:.2f}.png'.format(psnr_noise))
                img_out = Image.fromarray((img_out * 255).astype('uint8'))
                img_out.save(out_dir + filename + '_psnr{:.2f}.png'.format(psnr_denoised))

        psnr_avg = sum(psnr) / len(psnr)
        ssim_avg = sum(ssim) / len(ssim)
        name.append('Average')
        psnr.append(psnr_avg)
        ssim.append(ssim_avg)
        print('Average PSNR = {0:.4f}, SSIM = {1:.4f}'.format(psnr_avg, ssim_avg))
    else:
        # ----------------------------------------------------------------------#
        print("Test on Real Frame !")
        save_dir = 'results/real'
        multi_GPU = True
        # ----------------------------------------------------------------------#
        test_dir = './data/Test/Real/'
        WEIGHT_PATH = './checkpoints/weights-50-28.8004-0.0013.hdf5'
        if multi_GPU:
            model = NUCNN()
            print('Using Multi GPUs !')
            model = multi_gpu_model(model, gpus=2)
            model.load_weights(WEIGHT_PATH)
        else:
            model = NUCNN()
            model.load_weights(WEIGHT_PATH)
        print('Start to test on {}'.format(test_dir))
        out_dir = save_dir + '/' + test_dir.split('/')[-1] + '/'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        name = []
        print('Start Test')
        file_list = os.listdir(test_dir)
        for file in file_list:
            # read image
            img_clean = np.array(Image.open(test_dir + file), dtype='float32') / 255.0
            # img_test = img_clean + np.random.normal(0, sigma/255.0, img_clean.shape)
            img_test = img_clean.astype('float32')
            # predict
            x_test = img_test.reshape(1, img_test.shape[0], img_test.shape[1], 1)
            y_predict = model.predict(x_test)
            # calculate numeric metrics
            img_out = y_predict.reshape(img_clean.shape)
            img_out = np.clip(img_out, 0, 1)
            filename = file  # get the name of image file
            name.append(filename)

            img_out = Image.fromarray((img_out * 255).astype('uint8'))
            img_out.save(out_dir + filename)

        print('Test Over')
