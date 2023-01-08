# -*- coding: utf-8 -*-
"""
date:20210423
author: lhl
describe:多波段时空融合实验结果评价
function:
            rmse_me():计算单波段和总的均方根误差
            ssim_me():计算单波段和总的结构相似
            cc_me():计算单波段和总的相关系数
            ergas_me():计算总全局相对误差
            sam_me():计算光谱角制图
"""
import gdal
import skimage.metrics as sm
import numpy as np
import os,sys
sys.path.append(os.path.dirname(__file__)+os.sep+'../')

from scipy.stats import pearsonr
from numpy.linalg import norm

np.seterr(divide='ignore',invalid='ignore')

# #设置log.debug()写入文件
# logpath= r"E:\codes\stf\logs"
# # logpath = "/home1/hllu/codes/stf/logs"
# logfilenaem="testtarget.log"
# logger = utilme.log_save(logpath,logfilenaem )

def rmse_me(pre, targ):
    print("------------------------------")
    print("start compute rmse")
    # # print("pre shape and targ shape is "+str(pre.shape)+"|"+str(targ.shape))
    # logger.debug("------------------------------")
    # logger.debug("start compute rmse")
    # logger.debug("pre shape and targ shape is "+str(pre.shape)+"|"+str(targ.shape))
    if len(pre.shape) >= 3:
        c, w, h = pre.shape
    else:
        w, h, c = (pre.shape),1

    if c == 1:
        print("just one band")
        # logger.debug("just one band")
        result = sm.mean_squared_error(pre, targ)
        result = np.sqrt(result)
        print("rmse all is :", result)
        # logger.debug("rmse all is :"+str(result))
    else:
        for i in range(c):
            result = sm.mean_squared_error(pre[i], targ[i])
            result = np.sqrt(result)
            print("rmse b"+str(i+1)+" is "+str(result))
            # logger.debug("rmse b"+str(i+1)+" is "+str(result))
        result = sm.mean_squared_error(pre, targ)
        result = np.sqrt(result)
        print("rmse all is :", result)
        # logger.debug("rmse all is :"+str(result))

    return


def ssim_me(pre, targ):
    print("------------------------------")
    print("start compute ssim")
    # print("pre shape and targ shape is "+str(pre.shape)+"|"+str(targ.shape))
    # logger.debug("------------------------------")
    # logger.debug("start compute ssim")
    # logger.debug("pre shape and targ shape is "+str(pre.shape)+"|"+str(targ.shape))
    c = 1
    if len(pre.shape) >= 3:
        c, w, h = pre.shape


    if c == 1:
        print("just one band")
        # logger.debug("just one band")
        result = sm.structural_similarity(pre[0], targ[0], data_range=1)
        print("ssim all is :", result)
        # logger.debug("ssim all is :"+str(result))
    else:
        for i in range(c):
            result = sm.structural_similarity(pre[i], targ[i], data_range=1)
            print("ssim b"+str(i+1)+" is "+str(result))
            # logger.debug("ssim b"+str(i+1)+" is "+str(result))

        pre = pre.transpose(1, 2, 0)
        targ = targ.transpose(1, 2, 0)
        # result = sm.structural_similarity(pre, targ, data_range=1, multichannel=True)
        result = sm.structural_similarity(pre, targ, data_range=1, channel_axis=-1)
        print("ssim all is :", result)
        # logger.debug("ssim all is :"+str(result))
    return


def cc_me(pre, targ):
    print("------------------------------")
    print("start compute cc")
    # print("pre shape and targ shape is "+str(pre.shape)+"|"+str(targ.shape))
    # logger.debug("------------------------------")
    # logger.debug("start compute cc")
    # logger.debug("pre shape and targ shape is "+str(pre.shape)+"|"+str(targ.shape))
    if len(pre.shape) >= 3:
        c, w, h = pre.shape
    else:
        w, h, c = (pre.shape),1

    if c == 1:
        print("just one band")
        # logger.debug("just one band")
        pre1 = pre[0].reshape((w * h), order='C')
        targ1 = targ[0].reshape((w * h), order='C')
        result = pearsonr(pre1, targ1)[0]
        print("cc all is :", result)
        # logger.debug("cc all is :"+str(result))
    else:
        sum=0
        for i in range(c):
            pre1 = pre[i].reshape((w * h), order='C')
            targ1 = targ[i].reshape((w * h), order='C')
            result = pearsonr(pre1, targ1)[0]
            print("cc b"+str(i+1)+" is "+str(result))
            # logger.debug("cc b"+str(i+1)+" is "+str(result))
            sum=sum+result

        result = sum/c
        print("cc all is :", result)
        # logger.debug("cc all is :"+str(result))

    return


def ergas_me(pre, targ, ratio=0.03):

    print("------------------------------")
    print("start compute ergas")
    # print("pre shape and targ shape is "+str(pre.shape)+"|"+str(targ.shape))
    # logger.debug("------------------------------")
    # logger.debug("start compute ergas")
    # logger.debug("pre shape and targ shape is "+str(pre.shape)+"|"+str(targ.shape))
    c , w , h = pre.shape
    targ2 = targ.transpose(1, 2, 0)
    print(pre.shape)
    sum = 0.0
    for i in range(c):
        result = sm.mean_squared_error(pre[i], targ[i])
        result = np.sqrt(result)
        sum += result**2 / np.mean(targ2[:, :, i])**2
    re =  100 * ratio * np.sqrt(sum/c)
    print("ergas is :", re)
    # logger.debug("ergas is :" + str(re))
    return


def sam_me(pre, targ):
    print("------------------------------")
    print("start compute sam")

    # logger.debug("------------------------------")
    # logger.debug("start compute sam")

    pre1 = pre.transpose(1, 2, 0)
    targ1 = targ.transpose(1, 2, 0)
    # print(img2data.shape)
    assert pre1.ndim ==3 and pre1.shape == targ1.shape
    dot_sum = np.sum(pre1*targ1, axis=2)
    norm_true = norm(pre1, axis=2)
    norm_pred = norm(targ1, axis=2)
    res = np.arccos(dot_sum/norm_pred/norm_true)
    is_nan = np.nonzero(np.isnan(res))
    for (x, y) in zip(is_nan[0], is_nan[1]):
        res[x, y] = 00
    result = np.mean(res)
    print("sam is :", result)
    # logger.debug("sam is :" + str(result))
    return


def test():
    #STARFM
    #CIA
    # x = r"E:\codes\dataset\res\STARFM\CIA\STARFM_L2001_290_17oct.tif"
    # x = r"E:\codes\dataset\res\STARFM\CIA\STARFM_L2001_306_02nov.tif"
    # x = r"E:\codes\dataset\res\STARFM\CIA\STARFM_L2001_313_09nov.tif"
    # x = r"E:\codes\dataset\res\STARFM\CIA\STARFM_L2001_329_25nov.tif"
    # x = r"E:\codes\dataset\res\STARFM\CIA\STARFM_L2001_338_04dec.tif"
    # x = r"E:\codes\dataset\res\STARFM\CIA\STARFM_L2002_005_05jan.tif"
    # x = r"E:\codes\dataset\res\STARFM\CIA\STARFM_L2002_012_12jan.tif"
    # x = r"E:\codes\dataset\res\STARFM\CIA\STARFM_L2002_044_13feb.tif"
    # x = r"E:\codes\dataset\res\STARFM\CIA\STARFM_L2002_053_22feb.tif"
    # x = r"E:\codes\dataset\res\STARFM\CIA\STARFM_L2002_069_10mar.tif"
    # x = r"E:\codes\dataset\res\STARFM\CIA\STARFM_L2002_076_17mar.tif"
    # x = r"E:\codes\dataset\res\STARFM\CIA\STARFM_L2002_092_02apr.tif"
    # x = r"E:\codes\dataset\res\STARFM\CIA\STARFM_L2002_101_11apr.tif"
    # x = r"E:\codes\dataset\res\STARFM\CIA\STARFM_L2002_108_18apr.tif"
    # x = r"E:\codes\dataset\res\STARFM\CIA\STARFM_L2002_117_27apr.tif"

    #LGC
    # x = r"E:\codes\dataset\res\STARFM\LGC\STARFM_L20040502_TM.tif"
    # x = r"E:\codes\dataset\res\STARFM\LGC\STARFM_L20040705_TM.tif"
    # x = r"E:\codes\dataset\res\STARFM\LGC\STARFM_L20040806_TM.tif"
    # x = r"E:\codes\dataset\res\STARFM\LGC\STARFM_L20040822_TM.tif"
    # x = r"E:\codes\dataset\res\STARFM\LGC\STARFM_L20041025_TM.tif"
    # x = r"E:\codes\dataset\res\STARFM\LGC\STARFM_L20041126_TM.tif"
    # x = r"E:\codes\dataset\res\STARFM\LGC\STARFM_L20041212_TM.tif"
    # x = r"E:\codes\dataset\res\STARFM\LGC\STARFM_L20041228_TM.tif"
    # x = r"E:\codes\dataset\res\STARFM\LGC\STARFM_L20050113_TM.tif"
    # x = r"E:\codes\dataset\res\STARFM\LGC\STARFM_L20050129_TM.tif"
    # x = r"E:\codes\dataset\res\STARFM\LGC\STARFM_L20050214_TM.tif"
    # x = r"E:\codes\dataset\res\STARFM\LGC\STARFM_L20050302_TM.tif"

    #FSDAF
    #CIA
    # x = r"E:\codes\dataset\res\FSDAF\CIA\FSDAF_L2001_290_17oct.tif"
    # x = r"E:\codes\dataset\res\FSDAF\CIA\FSDAF_L2001_306_02nov.tif"
    # x = r"E:\codes\dataset\res\FSDAF\CIA\FSDAF_L2001_313_09nov.tif"
    # x = r"E:\codes\dataset\res\FSDAF\CIA\FSDAF_L2001_329_25nov.tif"
    # x = r"E:\codes\dataset\res\FSDAF\CIA\FSDAF_L2001_338_04dec.tif"
    # x = r"E:\codes\dataset\res\FSDAF\CIA\FSDAF_L2002_005_05jan.tif"
    # x = r"E:\codes\dataset\res\FSDAF\CIA\FSDAF_L2002_012_12jan.tif"
    # x = r"E:\codes\dataset\res\FSDAF\CIA\FSDAF_L2002_044_13feb.tif"
    # x = r"E:\codes\dataset\res\FSDAF\CIA\FSDAF_L2002_053_22feb.tif"
    # x = r"E:\codes\dataset\res\FSDAF\CIA\FSDAF_L2002_069_10mar.tif"
    # x = r"E:\codes\dataset\res\FSDAF\CIA\FSDAF_L2002_076_17mar.tif"
    # x = r"E:\codes\dataset\res\FSDAF\CIA\FSDAF_L2002_092_02apr.tif"
    # x = r"E:\codes\dataset\res\FSDAF\CIA\FSDAF_L2002_101_11apr.tif"
    # x = r"E:\codes\dataset\res\FSDAF\CIA\FSDAF_L2002_108_18apr.tif"
    # x = r"E:\codes\dataset\res\FSDAF\CIA\FSDAF_L2002_117_27apr.tif"

    #LGC
    # x = r"E:\codes\dataset\res\FSDAF\LGC\FSDAF_L20040502_TM.tif"
    # x = r"E:\codes\dataset\res\FSDAF\LGC\FSDAF_L20040705_TM.tif"
    # x = r"E:\codes\dataset\res\FSDAF\LGC\FSDAF_L20040806_TM.tif"
    # x = r"E:\codes\dataset\res\FSDAF\LGC\FSDAF_L20040822_TM.tif"
    # x = r"E:\codes\dataset\res\FSDAF\LGC\FSDAF_L20041025_TM.tif"
    # x = r"E:\codes\dataset\res\FSDAF\LGC\FSDAF_L20041126_TM.tif"
    # x = r"E:\codes\dataset\res\FSDAF\LGC\FSDAF_L20041212_TM.tif"
    # x = r"E:\codes\dataset\res\FSDAF\LGC\FSDAF_L20041228_TM.tif"
    # x = r"E:\codes\dataset\res\FSDAF\LGC\FSDAF_L20050113_TM.tif"
    # x = r"E:\codes\dataset\res\FSDAF\LGC\FSDAF_L20050129_TM.tif"
    # x = r"E:\codes\dataset\res\FSDAF\LGC\FSDAF_L20050214_TM.tif"
    # x = r"E:\codes\dataset\res\FSDAF\LGC\FSDAF_L20050302_TM.tif"

    #stfnet
    #CIA
    # x = r"E:\codes\dataset\res\stfnet\CIA\stfnet_L2001_290_17oct.tif"
    # x = r"E:\codes\dataset\res\stfnet\CIA\stfnet_L2001_306_02nov.tif"
    # x = r"E:\codes\dataset\res\stfnet\CIA\stfnet_L2001_313_09nov.tif"
    # x = r"E:\codes\dataset\res\stfnet\CIA\stfnet_L2001_329_25nov.tif"
    # x = r"E:\codes\dataset\res\stfnet\CIA\stfnet_L2001_338_04dec.tif"
    # x = r"E:\codes\dataset\res\stfnet\CIA\stfnet_L2002_005_05jan.tif"
    # x = r"E:\codes\dataset\res\stfnet\CIA\stfnet_L2002_012_12jan.tif"
    # x = r"E:\codes\dataset\res\stfnet\CIA\stfnet_L2002_044_13feb.tif"
    # x = r"E:\codes\dataset\res\stfnet\CIA\stfnet_L2002_053_22feb.tif"
    # x = r"E:\codes\dataset\res\stfnet\CIA\stfnet_L2002_069_10mar.tif"
    # x = r"E:\codes\dataset\res\stfnet\CIA\stfnet_L2002_076_17mar.tif"
    # x = r"E:\codes\dataset\res\stfnet\CIA\stfnet_L2002_092_02apr.tif"
    # x = r"E:\codes\dataset\res\stfnet\CIA\stfnet_L2002_101_11apr.tif"
    # x = r"E:\codes\dataset\res\stfnet\CIA\stfnet_L2002_108_18apr.tif"
    # x = r"E:\codes\dataset\res\stfnet\CIA\stfnet_L2002_117_27apr.tif"

    #LGC
    # x = r"E:\codes\dataset\res\stfnet\LGC\stfnet_L20040502_TM.tif"
    # x = r"E:\codes\dataset\res\stfnet\LGC\stfnet_L20040705_TM.tif"
    # x = r"E:\codes\dataset\res\stfnet\LGC\stfnet_L20040806_TM.tif"
    # x = r"E:\codes\dataset\res\stfnet\LGC\stfnet_L20040822_TM.tif"
    # x = r"E:\codes\dataset\res\stfnet\LGC\stfnet_L20041025_TM.tif"
    # x = r"E:\codes\dataset\res\stfnet\LGC\stfnet_L20041126_TM.tif"
    # x = r"E:\codes\dataset\res\stfnet\LGC\stfnet_L20041212_TM.tif"
    # x = r"E:\codes\dataset\res\stfnet\LGC\stfnet_L20041228_TM.tif"
    # x = r"E:\codes\dataset\res\stfnet\LGC\stfnet_L20050113_TM.tif"
    # x = r"E:\codes\dataset\res\stfnet\LGC\stfnet_L20050129_TM.tif"
    # x = r"E:\codes\dataset\res\stfnet\LGC\stfnet_L20050214_TM.tif"
    # x = r"E:\codes\dataset\res\stfnet\LGC\stfnet_L20050302_TM.tif"

    #stfmlp
    #CIA
    # x = r"D:\codes\dataset\res\stfmlp\CIA\stfmlp_L2001_290_17oct.tif"
    # x = r"D:\codes\dataset\res\stfmlp\CIA\stfmlp_L2001_306_02nov.tif"
    # x = r"D:\codes\dataset\res\stfmlp\CIA\stfmlp_L2001_313_09nov.tif"
    # x = r"D:\codes\dataset\res\stfmlp\CIA\stfmlp_L2001_329_25nov.tif"
    # x = r"D:\codes\dataset\res\stfmlp\CIA\stfmlp_L2001_338_04dec.tif"
    # x = r"D:\codes\dataset\res\stfmlp\CIA\stfmlp_L2002_005_05jan.tif"
    # x = r"D:\codes\dataset\res\stfmlp\CIA\stfmlp_L2002_012_12jan.tif"
    # x = r"D:\codes\dataset\res\stfmlp\CIA\stfmlp_L2002_044_13feb.tif"
    # x = r"D:\codes\dataset\res\stfmlp\CIA\stfmlp_L2002_053_22feb.tif"
    # x = r"D:\codes\dataset\res\stfmlp\CIA\stfmlp_L2002_069_10mar.tif"
    # x = r"D:\codes\dataset\res\stfmlp\CIA\stfmlp_L2002_076_17mar.tif"
    # x = r"D:\codes\dataset\res\stfmlp\CIA\stfmlp_L2002_092_02apr.tif"
    # x = r"D:\codes\dataset\res\stfmlp\CIA\stfmlp_L2002_101_11apr.tif"
    # x = r"D:\codes\dataset\res\stfmlp\CIA\stfmlp_L2002_108_18apr.tif"
    x = r"D:\codes\dataset\res\stfmlp\CIA\stfmlp_L2002_117_27apr.tif"

    #LGC
    # x = r"D:\codes\dataset\res\stfmlp\LGC\stfmlp_L20040502_TM.tif"
    # x = r"D:\codes\dataset\res\stfmlp\LGC\stfmlp_L20040705_TM.tif"
    # x = r"D:\codes\dataset\res\stfmlp\LGC\stfmlp_L20040806_TM.tif"
    # x = r"D:\codes\dataset\res\stfmlp\LGC\stfmlp_L20040822_TM.tif"
    # x = r"D:\codes\dataset\res\stfmlp\LGC\stfmlp_L20041025_TM.tif"
    # x = r"D:\codes\dataset\res\stfmlp\LGC\stfmlp_L20041126_TM.tif"
    # x = r"D:\codes\dataset\res\stfmlp\LGC\stfmlp_L20041212_TM.tif"
    # x = r"D:\codes\dataset\res\stfmlp\LGC\stfmlp_L20041228_TM.tif"
    # x = r"D:\codes\dataset\res\stfmlp\LGC\stfmlp_L20050113_TM.tif"
    # x = r"D:\codes\dataset\res\stfmlp\LGC\stfmlp_L20050129_TM.tif"
    # x = r"D:\codes\dataset\res\stfmlp\LGC\stfmlp_L20050214_TM.tif"
    # x = r"D:\codes\dataset\res\stfmlp\LGC\stfmlp_L20050302_TM.tif"

    #CIA TARGET
    # target = r"D:\codes\dataset\CIA\train_set\L2001_290_17oct.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2001_306_02nov.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2001_313_09nov.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2001_329_25nov.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2001_338_04dec.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_005_05jan.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_012_12jan.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_044_13feb.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_053_22feb.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_069_10mar.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_076_17mar.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_092_02apr.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_101_11apr.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_108_18apr.tif"
    target = r"D:\codes\dataset\CIA\train_set\L2002_117_27apr.tif"

    #LGC TARGET
    # target = r"D:\codes\dataset\LGC\train_set\L20040502_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20040705_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20040806_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20040822_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20041025_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20041126_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20041212_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20041228_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20050113_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20050129_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20050214_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20050302_TM.tif"


    channels = 6
    start = 0
    img_x = gdal.Open(x)

    img_y = gdal.Open(target)

    pre = img_x.ReadAsArray()/ 10000
    targ = img_y.ReadAsArray()/ 10000

    # if pre.shape[1]!=targ.shape[1]:
    #     targ = targ[:,:pre.shape[1],:pre.shape[2]]

    if channels==6:
        pre = pre[0:,:,:]
        targ = targ[0:,:,:]
    elif start == 0:
        pre = pre[0:3,:,:]
        targ = targ[0:3,:,:]
    else:
        pre = pre[3:,:,:]
        targ = targ[3:,:,:]

    #由于李军老师提出的数据集实验中，反射率存储时乘以255倍，所以除以255进行还原


    # pre = img_x.ReadAsArray() /255
    # # pre2 = img_xmodis.ReadAsArray()/ 255
    # # pre = pre1 - pre2
    # targ = img_y.ReadAsArray()/255
    # 计算rmsex
    # pre = img_x.ReadAsArray() / 255
    # targ = img_y.ReadAsArray()/ 255
    rmse_me(pre, targ)

    # 计算cc
    cc_me(pre, targ)

    # 计算ssim
    ssim_me(pre, targ)

    # 计算 ergas
    ergas_me(pre, targ)

    #计算 sam
    sam_me(pre, targ)



def test_re1():

    #stfmlp_re1
    #CIA
    # x = r"D:\codes\dataset\res\re1_result\stfmlp_L2001_290_17oct.tif"
    # x = r"D:\codes\dataset\res\re1_result\stfmlp_L2001_306_02nov.tif"
    # x = r"D:\codes\dataset\res\re1_result\stfmlp_L2001_313_09nov.tif"
    # x = r"D:\codes\dataset\res\re1_result\stfmlp_L2001_329_25nov.tif"
    # x = r"D:\codes\dataset\res\re1_result\stfmlp_L2001_338_04dec.tif"
    # x = r"D:\codes\dataset\res\re1_result\stfmlp_L2002_005_05jan.tif"
    # x = r"D:\codes\dataset\res\re1_result\stfmlp_L2002_012_12jan.tif"
    # x = r"D:\codes\dataset\res\re1_result\stfmlp_L2002_044_13feb.tif"
    # x = r"D:\codes\dataset\res\re1_result\stfmlp_L2002_053_22feb.tif"
    # x = r"D:\codes\dataset\res\re1_result\stfmlp_L2002_069_10mar.tif"
    # x = r"D:\codes\dataset\res\re1_result\stfmlp_L2002_076_17mar.tif"
    # x = r"D:\codes\dataset\res\re1_result\stfmlp_L2002_092_02apr.tif"
    # x = r"D:\codes\dataset\res\re1_result\stfmlp_L2002_101_11apr.tif"
    # x = r"D:\codes\dataset\res\re1_result\stfmlp_L2002_108_18apr.tif"
    # x = r"D:\codes\dataset\res\re1_result\stfmlp_L2002_117_27apr.tif"

   



    #CIA TARGET
    # target = r"D:\codes\dataset\CIA\train_set\L2001_290_17oct.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2001_306_02nov.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2001_313_09nov.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2001_329_25nov.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2001_338_04dec.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_005_05jan.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_012_12jan.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_044_13feb.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_053_22feb.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_069_10mar.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_076_17mar.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_092_02apr.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_101_11apr.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_108_18apr.tif"
    # target = r"D:\codes\dataset\CIA\train_set\L2002_117_27apr.tif"




    #LGC

    # x = r"D:\codes\dataset\res\re1_result\lgc\stfmlp_L20040502_TM.tif"
    # x = r"D:\codes\dataset\res\re1_result\lgc\stfmlp_L20040705_TM.tif"
    # x = r"D:\codes\dataset\res\re1_result\lgc\stfmlp_L20040806_TM.tif"
    # x = r"D:\codes\dataset\res\re1_result\lgc\stfmlp_L20040822_TM.tif"
    # x = r"D:\codes\dataset\res\re1_result\lgc\stfmlp_L20041025_TM.tif"
    # x = r"D:\codes\dataset\res\re1_result\lgc\stfmlp_L20041126_TM.tif"
    # x = r"D:\codes\dataset\res\re1_result\lgc\stfmlp_L20041212_TM.tif"
    # x = r"D:\codes\dataset\res\re1_result\lgc\stfmlp_L20041228_TM.tif"
    # x = r"D:\codes\dataset\res\re1_result\lgc\stfmlp_L20050113_TM.tif"
    # x = r"D:\codes\dataset\res\re1_result\lgc\stfmlp_L20050129_TM.tif"
    # x = r"D:\codes\dataset\res\re1_result\lgc\stfmlp_L20050214_TM.tif"
    x = r"D:\codes\dataset\res\re1_result\lgc\stfmlp_L20050302_TM.tif"
    #LGC TARGET
    # target = r"D:\codes\dataset\LGC\train_set\L20040502_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20040705_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20040806_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20040822_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20041025_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20041126_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20041212_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20041228_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20050113_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20050129_TM.tif"
    # target = r"D:\codes\dataset\LGC\train_set\L20050214_TM.tif"
    target = r"D:\codes\dataset\LGC\train_set\L20050302_TM.tif"






    channels = 6
    start = 0
    img_x = gdal.Open(x)

    img_y = gdal.Open(target)

    pre = img_x.ReadAsArray()/ 10000
    targ = img_y.ReadAsArray()/ 10000

    # if pre.shape[1]!=targ.shape[1]:
    #     targ = targ[:,:pre.shape[1],:pre.shape[2]]

    if channels==6:
        pre = pre[0:,:,:]
        targ = targ[0:,:,:]
    elif start == 0:
        pre = pre[0:3,:,:]
        targ = targ[0:3,:,:]
    else:
        pre = pre[3:,:,:]
        targ = targ[3:,:,:]

    #由于李军老师提出的数据集实验中，反射率存储时乘以255倍，所以除以255进行还原


    # pre = img_x.ReadAsArray() /255
    # # pre2 = img_xmodis.ReadAsArray()/ 255
    # # pre = pre1 - pre2
    # targ = img_y.ReadAsArray()/255
    # 计算rmsex
    # pre = img_x.ReadAsArray() / 255
    # targ = img_y.ReadAsArray()/ 255
    rmse_me(pre, targ)

    # 计算cc
    cc_me(pre, targ)

    # 计算ssim
    ssim_me(pre, targ)

    # 计算 ergas
    ergas_me(pre, targ)

    #计算 sam
    sam_me(pre, targ)

if __name__ == "__main__":
    # test()
    test_re1()