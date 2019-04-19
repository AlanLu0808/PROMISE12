"""
Created on Fri Oct 27 2017
@author: Inom Mirzaev
"""

from __future__ import division, print_function

import cv2
import matplotlib.pyplot as plt
from keras.optimizers import Adam

plt.switch_backend('agg')
import numpy as np
import os
import SimpleITK as sitk

from losses import make_loss, dice_coef
from models.models import get_model
from sklearn.model_selection import KFold
import pandas as pd
import keras.backend as K

# train_mhd_path = '/mnt/data1/jzb/data/PROMISE12/mhd_data/train/'
# test_mhd_path = '/mnt/data1/jzb/data/PROMISE12/mhd_data/test/'
# npy_data_basepath = '/mnt/data1/jzb/data/PROMISE12/npy_data/'

network = 'unet_resnet_18'

# weights_path = '../weights/unet_resnet_18_stage_2_dice_loss_ohem_fold_'
# weights_path = '../weights/unet_resnet_18_stage_1_1_dice_loss_fold_'
weights_path = 'weights/ori_baseline'
learning_rate = 0.0001
loss_function = 'bce_dice'
# loss_function = 'lovasz'

train_mhd_path = '/mnt/data/datasets/PROMISE12/train/'
test_mhd_path = '/mnt/data/datasets/PROMISE12/test/'
train_npy_path = '/mnt/data/project/ljg/pycharmProject/seg/promise12_pretrain_unet/promise12_pretrain_unet/codes/train_data/'
test_npy_path = '/mnt/data/project/ljg/pycharmProject/seg/promise12_pretrain_unet/promise12_pretrain_unet/codes/test_data/'

# predict_test_mhd_path = '/mnt/data1/jzb/data/PROMISE12/mhd_data/pred_test_unet_resnet_18_stage_1_1_dice_loss_TTA_fold_all/'

predict_test_mhd_path = 'pred_baseline/'

predict_train_mhd_path = 'pred_baseline_train/'


def resize_pred_to_val(y_pred, shape):
    row = shape[1]
    col = shape[2]
    resized_pred = np.zeros(shape)
    for mm in range(len(y_pred)):
        resized_pred[mm, :, :] = cv2.resize(y_pred[mm, :, :, 0], (row, col), interpolation=cv2.INTER_NEAREST)
    return resized_pred


def dice_coef_compute(y_true, y_pred, smooth=1.0):

    y_true_f = y_true
    y_pred_f = y_pred
    y_true_f.flatten()
    y_pred_f.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true*y_true) + np.sum(y_pred*y_pred)
    return (2. * intersection + smooth) / (union + smooth)


def get_models(img_rows, img_cols, fold):
    model, process = get_model(network=network, input_shape=(img_rows, img_cols, 1),
                               freeze_encoder=False)
    model.load_weights(weights_path + str(fold) + '.hdf5')
    model.compile(optimizer=Adam(lr=learning_rate), loss=make_loss(loss_name=loss_function), metrics=[dice_coef])
    return model, process

def load_data_train(kfold=None):
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
    X_train = X_train[:, 32:32+192, 32:32+192, :]
    y_train = y_train[:, 32:32+192, 32:32+192, :]
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
    X_val = X_val[:, 32:32+192, 32:32+192, :]
    y_val = y_val[:, 32:32+192, 32:32+192, :]
    print(X_val.shape, y_val.shape)
    return X_train, y_train, X_val, y_val, df_train, df_val, train_index, val_index

def load_data_test():
    fileList = os.listdir(test_npy_path)
    X_List = list(filter(lambda x: '.npy' in x and 'segm' not in x, fileList))
    X_List.sort()
    X_test = []
    for filename in X_List:
        # print(filename)
        image = np.load(test_npy_path + filename)
        X_test.append(image)
        # break

    X_test = np.concatenate(X_test, axis=0)
    X_test = np.array(X_test)
    print(X_test.shape)
    X_test = X_test[:, 16:16+224, 16:16+224, :]
    print(X_test.shape)
    return X_test

def predict_train(folder=train_mhd_path, dest=predict_train_mhd_path, fold=0):

    if not os.path.isdir(dest):
        os.mkdir(dest)
    print('fold = {}'.format(fold))
    print('begin load data')
    X_train, y_train, X_val, y_val, df_train, df_val, train_index, val_index = load_data_train(fold)
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    # (1250, 256, 256, 1) (1250, 256, 256, 1) (127, 256, 256, 1) (127, 256, 256, 1)
    print('load data over')

    img_rows = X_train.shape[1]
    img_cols = img_rows

    model, process = get_models(img_rows, img_cols, fold)
    X_val = process(X_val)
    y_pred = model.predict(X_val, verbose=1, batch_size=32)
    # (1250, 256, 256, 1)
    print(y_pred.shape)

    padding_y_pred = np.zeros((y_pred.shape[0], 256, 256, 1))
    padding_y_pred[:, 32:32+192, 32:32+192, :] = y_pred
    print(padding_y_pred.shape)


    fileList = os.listdir(train_mhd_path)
    X_List = list(filter(lambda x: '.mhd' in x and 'segm' not in x, fileList))
    X_List.sort()
    # print(X_List)
    y_List = list(filter(lambda x: '.mhd' in x and 'segm' in x, fileList))
    y_List.sort()
    # print(y_List)
    dicts = {'X': X_List, 'y': y_List}
    df = pd.DataFrame(dicts)
    df_train_mhd = df.iloc[train_index]
    df_val_mhd = df.iloc[val_index]
    # print(df_train)
    # print(df_train_mhd)
    # print(df_val)
    # print(df_val_mhd)

    start_ind=0
    end_ind=0
    total = 0
    count = 0
    for filename in list(df_val_mhd['X']):
        print(filename)
        itkimage = sitk.ReadImage(folder+filename)
        img = sitk.GetArrayFromImage(itkimage)
        # print(img.shape) # (47, 512, 512)
        start_ind = end_ind
        end_ind +=len(img)

        # mask = y_pred[start_ind:end_ind]
        mask = padding_y_pred[start_ind:end_ind]
        print(mask.shape)
        mask = (mask > 0.5).astype(np.float32)

        # #to compute dice for train npy and print the dice
        # dice = dice_coef_compute(y_val[start_ind:end_ind].astype(np.float32), mask, smooth=1.0)
        # print(filename, ' dice_coef is ', dice)
        # total = total + dice

        pred = resize_pred_to_val(mask, img.shape )
        # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        pred = np.squeeze(pred)
        print(pred.shape) # (47, 512, 512)

        mask = sitk.GetImageFromArray( pred)
        mask.SetOrigin( itkimage.GetOrigin() )
        mask.SetDirection( itkimage.GetDirection() )
        mask.SetSpacing( itkimage.GetSpacing() )

        sitk.WriteImage(mask, dest+'/'+filename[:-4]+'_segmentation.mhd')
        # break
    #     count+=1
    # print("total", total/count)


def predict_test(folder=test_mhd_path, dest=predict_test_mhd_path, fold=None):
    if not os.path.isdir(dest):
        os.mkdir(dest)
    print('begin load data')
    X_test = load_data_test()
    print(X_test.shape)
    # (1250, 256, 256, 1)
    print('load data over')

    img_rows = X_test.shape[1]
    img_cols = img_rows

    model, process = get_models(img_rows, img_cols, fold)
    X_test = process(X_test)
    y_pred = model.predict(X_test, verbose=1, batch_size=32)
    # (892, 192, 192, 1)
    print(y_pred.shape)
    padding_y_pred = np.zeros((y_pred.shape[0], 256, 256, 1))
    padding_y_pred[:, 16:16+224, 16:16+224, :] = y_pred
    print(padding_y_pred.shape)
    X_List = os.listdir(test_mhd_path)
    X_List = list(filter(lambda x: '.mhd' in x, X_List))
    X_List.sort()
    print(X_List)
    start_ind = 0
    end_ind = 0
    total = 0
    count = 0
    for filename in X_List:
        print(filename)
        itkimage = sitk.ReadImage(folder + filename)
        img = sitk.GetArrayFromImage(itkimage)
        print(img.shape) # (47, 512, 512)
        start_ind = end_ind
        end_ind += len(img)

        mask = padding_y_pred[start_ind:end_ind]
        print(mask.shape)
        mask = (mask > 0.5).astype(np.float32)

        pred = resize_pred_to_val(mask, img.shape)

        print(pred.shape)
        # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        pred = np.squeeze(pred)

        # # 将图像类型强制转换成其他类型
        # castImageFilter = sitk.CastImageFilter()
        # # 将图像转换成像素类型
        # castImageFilter.SetOutputPixelType(sitk.sitkInt16)
        # print(itkimage.GetPixelIDTypeAsString())16-bit signed integer

        mask = sitk.GetImageFromArray( pred)
        mask.SetOrigin( itkimage.GetOrigin() )
        mask.SetDirection( itkimage.GetDirection() )
        mask.SetSpacing( itkimage.GetSpacing() )
        # # 使用给定参数执行过滤器
        # mask = castImageFilter.Execute(mask)

        sitk.WriteImage(mask, dest+'/'+filename[:-4]+'_segmentation.mhd')
        # break

def predict_test_TTA(fold=None):

    print('begin load data')
    X_test = load_data_test()
    print(X_test.shape)
    # (1250, 256, 256, 1)
    print('load data over')

    img_rows = X_test.shape[1]
    img_cols = img_rows

    model, process = get_models(img_rows, img_cols, fold)
    # TTA with horizontal flip
    X_test_origibal = X_test
    X_test_horizontal_flip = X_test[:, :, ::-1, :]
    X_test_vertical_flip = X_test[:, ::-1, :, :]
    X_test_origibal = process(X_test_origibal)
    X_test_horizontal_flip = process(X_test_horizontal_flip)
    X_test_vertical_flip = process(X_test_vertical_flip)

    print('**************predict original image****************')
    y_pred_original = model.predict(X_test_origibal, verbose=1, batch_size=32)
    # (892, 192, 192, 1)
    print(y_pred_original.shape)
    padding_y_pred_original = np.zeros((y_pred_original.shape[0], 256, 256, 1))
    padding_y_pred_original[:, 16:16+224, 16:16+224, :] = y_pred_original
    print(padding_y_pred_original.shape)

    print('**************predict horizontal_flip image****************')
    y_pred_horizontal_flip = model.predict(X_test_horizontal_flip, verbose=1, batch_size=32)
    # (892, 192, 192, 1)
    print(y_pred_horizontal_flip.shape)
    padding_y_pred_horizontal_flip = np.zeros((y_pred_horizontal_flip.shape[0], 256, 256, 1))
    padding_y_pred_horizontal_flip[:, 16:16+224, 16:16+224, :] = y_pred_horizontal_flip
    print(padding_y_pred_horizontal_flip.shape)

    y_pred_final = (padding_y_pred_original + padding_y_pred_horizontal_flip[:, :, ::-1, :])/2.
    return y_pred_final


def pred_test_result_to_mhd(folder=None, dest=None, y_pred=None):
    if not os.path.isdir(dest):
        os.mkdir(dest)
    X_List = os.listdir(test_mhd_path)
    X_List = list(filter(lambda x: '.mhd' in x, X_List))
    X_List.sort()
    print(X_List)
    start_ind = 0
    end_ind = 0
    total = 0
    count = 0
    for filename in X_List:
        print(filename)
        itkimage = sitk.ReadImage(folder + filename)
        img = sitk.GetArrayFromImage(itkimage)
        print(img.shape)  # (47, 512, 512)
        start_ind = end_ind
        end_ind += len(img)

        mask = y_pred[start_ind:end_ind]
        print(mask.shape)
        mask = (mask > 0.5).astype(np.float32)

        pred = resize_pred_to_val(mask, img.shape)

        print(pred.shape)
        # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        pred = np.squeeze(pred)

        # # 将图像类型强制转换成其他类型
        # castImageFilter = sitk.CastImageFilter()
        # # 将图像转换成像素类型
        # castImageFilter.SetOutputPixelType(sitk.sitkInt16)
        # print(itkimage.GetPixelIDTypeAsString())16-bit signed integer

        mask = sitk.GetImageFromArray(pred)
        mask.SetOrigin(itkimage.GetOrigin())
        mask.SetDirection(itkimage.GetDirection())
        mask.SetSpacing(itkimage.GetSpacing())
        # # 使用给定参数执行过滤器
        # mask = castImageFilter.Execute(mask)

        sitk.WriteImage(mask, dest + '/' + filename[:-4] + '_segmentation.mhd')
        # break

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
if __name__=='__main__':
    kfolds = [0]
    # kfolds = [0, 1]
    y_pred_list = []
    for fold in kfolds:
        print('fold=', fold)
        K.clear_session()
        y_pred = predict_test_TTA(fold)
        y_pred_list.append(y_pred)

    y_pred_final = np.mean(np.array(y_pred_list), axis=0)
    print(np.array(y_pred_list).shape, y_pred_final.shape)
    pred_test_result_to_mhd(folder=test_mhd_path, dest=predict_test_mhd_path, y_pred=y_pred_final)


    # predict_test(fold=0)
    # predict_train(fold=0)
    # test = load_data_test()
    # model, process = get_model(network=network, input_shape=(192, 192, 1),
    #                            freeze_encoder=False)
    # model.load_weights(weights_path + str(0) + '.hdf5')
    # # model.compile(optimizer=Adam(lr=learning_rate), loss=make_loss(loss_name=loss_function), metrics=[dice_coef])
    # model.summary()
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)
