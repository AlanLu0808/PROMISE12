"""
Created on Fri Sep 15 17:18:38 2017

@author: Inom Mirzaev
"""

from __future__ import division, print_function

import os

# from itertools import izip
izip = zip
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
import keras.backend as K
# from metrics import dice_coef, dice_coef_loss
from augmenters import *

from generator import Generator
# from unet_pretrain_model import get_model
from losses import make_loss, dice_coef
# from callbacks.callbacks import get_callback
from models.models import get_model
from sklearn.model_selection import KFold
import pandas as pd
from autoplot import AutoPlot
import gc

train_mhd_path = '/mnt/data1/jzb/data/PROMISE12/mhd_data/train/'
test_mhd_path = '/mnt/data1/jzb/data/PROMISE12/mhd_data/test/'
train_npy_path = 'train_data/'
from callbacks.snapshot import SnapshotCallbackBuilder


# def focal_loss(gamma=2, alpha=0.75):
#     def focal_loss_fixed(y_true, y_pred):#with tensorflow
#         eps = 1e-12
#         y_pred=K.clip(y_pred, eps, 1.-eps)#improve the stability of the focal loss and see issues 1 for more information
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
#     return focal_loss_fixed


def load_data(kfold=None):
    fileList = os.listdir(train_npy_path)
    X_List = list(filter(lambda x: '.npy' in x and 'segm' not in x, fileList))
    X_List.sort()
    # print(X_List)
    y_List = list(filter(lambda x: '.npy' in x and 'segm' in x, fileList))
    y_List.sort()
    # print(y_List)
    dicts = {'X': X_List, 'y': y_List}
    df = pd.DataFrame(dicts)
    # print(df)
    seed = 2
    np.random.seed(seed)
    folder = KFold(n_splits=5, shuffle=True, random_state=seed)
    train_index, val_index = list(folder.split(df))[kfold]
    df_train = df.iloc[train_index]
    df_val = df.iloc[val_index]
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    for filename in list(df_train['X']):
        # print(filename)
        image = np.load(train_npy_path + filename)
        X_train.append(image)
        mask = np.load(train_npy_path + filename.split('.')[0] + '_segmentation.npy')
        y_train.append(mask)
        # break
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = X_train[:, 16:16+192, 16:16+192, :]
    y_train = y_train[:, 16:16+192, 16:16+192, :]
    print(X_train.shape, y_train.shape)

    for filename in list(df_val['X']):
        # print(filename)
        image = np.load(train_npy_path + filename)
        X_val.append(image)
        mask = np.load(train_npy_path + filename.split('.')[0] + '_segmentation.npy')
        y_val.append(mask)
        # break
    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_val = X_val[:, 16:16+192, 16:16+192, :]
    y_val = y_val[:, 16:16+192, 16:16+192, :]
    print(X_val.shape, y_val.shape)
    return X_train, y_train, X_val, y_val

def get_callback(callback, fold=None, num_sample=None):
    if callback == 'reduce_lr':
        es_callback = EarlyStopping(monitor="val_dice_coef", patience=early_stop_patience, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef', factor=reduce_lr_factor,
                                      patience=reduce_lr_patience,
                                      min_lr=reduce_lr_min, verbose=1, mode='max')
        csv_logger = CSVLogger(filename=csv_path + str(fold) + '.csv', separator=',', append=True)
        auto_plot = AutoPlot(fig_path=fig_path, fold=fold)
        # SGDR_lr = SGDRScheduler(min_lr=1e-7, max_lr=1e-3, steps_per_epoch=np.ceil(num_sample/batch_size), lr_decay=0.9, cycle_length=30, mult_factor=1)

        callbacks = [es_callback, reduce_lr, auto_plot, csv_logger]
        # callbacks = [es_callback, auto_plot, csv_logger]
    elif callback == 'snapshot':
        snapshot = SnapshotCallbackBuilder(weights_path+str(fold)+'$$$$$$$$$$$$$$$.hdf5', epochs, n_snapshots, learning_rate)
        callbacks = snapshot.get_callbacks(model_prefix='snapshot', fold=fold)
    else:
        ValueError("Unknown callback")

    mc_callback_best = ModelCheckpoint(weights_path+str(fold)+'.hdf5', monitor='val_dice_coef', verbose=1, save_best_only=True,
                                       save_weights_only=True, mode='max')
    callbacks.append(mc_callback_best)

    return callbacks


def keras_fit_generator():

    kfolds = [0]
    # kfolds = [2, 3, 4]
    for fold in kfolds:

        K.clear_session()
        print('fold = {}'.format(fold))
        print('begin load data')
        X_train, y_train, X_val, y_val = load_data(fold)
        print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
        # (1250, 256, 256, 1) (1250, 256, 256, 1) (127, 256, 256, 1) (127, 256, 256, 1)
        print('load data over')


        model, process = get_model(network=network, input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), freeze_encoder=False)

        model.load_weights(pretrain_weight + str(fold) + '.hdf5')

        val_gen = Generator(X_val, y_val, batch_size=len(y_val), shuffle=True, aug=True, process=process)
        X_val_steps, y_val_steps = next(val_gen.generator)

        train_gen = Generator(X_train, y_train, batch_size=batch_size, shuffle=True, aug=True, process=process)

        model.compile(optimizer=Adam(lr=learning_rate), loss=make_loss(loss_name=loss_function), metrics=[dice_coef])

        model.summary()

        c_backs = get_callback(callback, fold, num_sample=len(X_train))

        model.fit_generator(
                            train_gen.generator,
                            steps_per_epoch=(len(X_train)//batch_size)*2,
                            epochs=epochs,
                            verbose=1,
                            shuffle=True,
                            validation_data=(X_val_steps, y_val_steps),
                            callbacks=c_backs,
                            use_multiprocessing=False)
        gc.collect()


# optimizer=Adam
epochs = 150
folds = 5
learning_rate = 0.00005
input_size = 256
resize_size = 192
batch_size = 32
# loss_function = 'lovasz'
loss_function = 'bce_dice'

callback = 'reduce_lr'
n_snapshots = 1
early_stop_patience = 31
reduce_lr_factor = 0.5
reduce_lr_patience = 10
reduce_lr_min = 1e-7
network = 'unet_resnet_18'

pretrain_weight = '../weights/unet_resnet_18_stage_2_dice_loss_sgdr_fold_'

weights_path = '../weights/unet_'
csv_path = '../csv_logger/resnet_'
fig_path = '../fig_save/resnet_'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


if __name__=='__main__':

    import time
    start = time.time()
    # K.clear_session()
    # load_data(kfold=0)
    # new_data_to_array(256, 256)

    # print('begin load data')
    # X_train, y_train, X_val, y_val = load_data()
    # print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    # print('load data over')
    #
    # train_gen = Generator(X_train, y_train, batch_size=32, shuffle=True, aug=True)
    # X, y = next(train_gen.generator)
    # print(X.shape, y.shape)

    keras_fit_generator()
    end = time.time()

    print('Elapsed time:', round((end-start)/60, 2))
