# -*- coding: utf-8 -*-
"""

    author: lhl

"""
import torch
import os
import numpy as np
import time
import random
import gdal
from v4_model import StfNet, Loss
from v6_dataset import MyDataset
import utilme
import argparse
import torch.optim as optim
from torch import  nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from itertools import chain
from tqdm.notebook import tqdm
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler

# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    np.random.seed(seed) # Numpy module.
    random.seed(seed) # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def main(i):
    # configpath =r"D:\codes\codes\stf\mlp\v8\v8_test_config.yaml"
    configpath ="/home/hllu/codes/stf/mlp/v8/v8_test_config.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=configpath, help="...")  # a.yaml中内容在文章开始给出
    args = parser.parse_args()
    configs = utilme.test_argparse(args)
    logpath = configs.logpath
    print("logpath is : %s"%logpath)
    version = configs.version
    # -----------------------------------------------------
    # times = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    timess = time.asctime(time.localtime(time.time()))
    print("#####start time: %s######"%timess)
    idx = i
    checkpoint_path = configs.checkpoint_path
    ckp_name = str(version) + "_idx_" + str(idx) + "_.pth"
    # ckp_name = str(version) + "_train_v2__idx_" + str(idx) + "_100.pth"
    checkpoint_path = os.path.join(checkpoint_path, ckp_name)
    # txtfile = configs.txtfile
    # checkpoints_name =configs.checkpoints_nameS
    # checkpoint_path = os.path.join(configs.logpath, checkpoints_name)


    batch_size = configs.batch_size
    patchrow = configs.img_size
    patchcol = configs.img_size
    channels = configs.outchannel

    overlap = 0
    delta = 0.2
    alpha = 0.0

    seed = configs.seed

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    seed_everything(seed)
    imglistdir = configs.imglistdir
    datalist = utilme.readdatasist(imglistdir)
    model = StfNet(configs).to(device)
    # print(model)
    if os.path.exists(checkpoint_path):
        # model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        # model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
        model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(checkpoint_path, map_location=torch.device(device)).items()})
        print("    Success to loading model dict from %s ....."%checkpoint_path)
    else:
        print("    Failed to load model dict  from %s ....."%checkpoint_path)
        return

    # print(model)
    # if torch.cuda.device_count() > 1:
    #     # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
    #     model = torch.nn.DataParallel(model, device_ids=[1])
    #     # model = torch.nn.DataParallel(model)
#-----------------------------------------------------
    resultname = "stfmlp_"+datalist[idx][1]
    save_path = os.path.join(logpath, resultname)
    traindata = MyDataset(configs.trainpath, datalist[idx], patchrow, patchcol,overlap=overlap)
    temdata = traindata.__getitem__(0)
    #修改这部分代码，此部分代表论文原文中取整幅图像差值的平均值，进行权重的计算。改成对每一个分块分别计算权重。所以改写到下面For循环之内。
    v12 = temdata["v12"]
    v23 = temdata["v23"]
    if (v23 - v12) > delta:
        alpha = 1.0
    elif (v12 - v23) > delta:
        alpha = 0.0
    else:
        alpha = (1/v12)/(1/v12+1/v23)
    # alpha = 1.0
    # mybate = v23 / (v12 + v23)
    print("delta:%s"%delta)
    print("v23-v12:%s"%(v23-v12))
    print("v12-v23:%s" % (v12 - v23))
    # print(mybate)
    print(alpha)
    realr = temdata['r']
    realc = temdata['c']
    row = temdata['paddr']
    col = temdata['paddc']
    img = torch.zeros((channels,row,col))
    train_loader = DataLoader(dataset=traindata, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        model.eval()
        for i, data in enumerate(train_loader):
            idxx = data["idx"].to(device)
            c1 = data["c1"].to(device)
            c2 = data["c2"].to(device)
            c3 = data["c3"].to(device)
            f1 = data["f1"].to(device)
            f3 = data["f3"].to(device)
            # label = data["f2"].to(device)
            # idxx = data['idx']

            # c12 = torch.abs(c2 / 10000 - c1 / 10000)
            # c23 = torch.abs(c3 / 10000 - c2 / 10000)
            #
            # sumc12 = torch.sum(c12)
            # sumc23 = torch.sum(c23)
            # v12 = sumc12 / (batch_size*channels * patchrow * patchrow)
            # v23 = sumc23 / (batch_size*channels * patchrow * patchrow)
            #
            # if (v23 - v12) > delta:
            #     alpha = 1.0
            # elif (v12 - v23) > delta:
            #     alpha = 0.0
            # else:
            #     alpha = (1/v12)/(1/v12+1/v23)

            # print("process %s/%s:random %s...." % (i, len(train_loader), idxx))
            c1 = c1.to(torch.float32) / 10000
            c2 = c2.to(torch.float32) / 10000
            c3 = c3.to(torch.float32) / 10000
            f1 = f1.to(torch.float32) / 10000
            f3 = f3.to(torch.float32) / 10000
            # label = label.to(torch.float32)/10000
            x1, x2, f12, f23 = model(c1, c2, c3, f1, f3)

            # f2 = 0.5 * (f1 + f12*alpha) +  0.5*(f3 - f23*(1-alpha))
            # tem = self.mybate * x4 + (1 - self.mybate) * x3
            # f12 = (1 - mybate) * f12
            # f23 = mybate * f23

            f2 = alpha * (f1 + f12) + (1-alpha) * (f3 - f23)
            f2 = torch.squeeze(f2, dim=0)
            for i, f2i in enumerate(f2):
                rnum = row / patchrow
                cnum = col / patchcol
                #banchsize>1时使用以下语句
                tem = idxx[i]
                # tem = idxx
                idr = int(tem // cnum)
                idc = int(tem % cnum)
                idrstart = patchrow * idr
                idrend = patchrow * idr + patchrow
                idcstart = patchcol * idc
                idcend = patchcol * idc + patchcol
                if (idrstart - overlap) >= 0:
                    idrstart -= overlap

                idrend += overlap
                if (idcstart - overlap) >= 0:
                    idcstart -= overlap
                idcend += overlap

                # 1 重叠区域取平均值
                # temp = img[:, idrstart:idrend, idcstart:idcend]
                # bb,h,w = f2.shape
                #
                # for bbi in range(bb):
                #     for hi in range(h):
                #         for wi in range(w):
                #             if temp[bbi,hi,wi] == 0:
                #                 temp[bbi,hi,wi] = f2[bbi,hi,wi]
                #             else:
                #                 temp[bbi,hi,wi] = (temp[bbi,hi,wi]+f2[bbi,hi,wi])/2
                # img[:, idrstart:idrend, idcstart:idcend] = temp
                # 2 无重叠或者重叠区域直接覆盖
                img[:, idrstart:idrend, idcstart:idcend] = f2i
    out = img[:, 0:realr, 0:realc].numpy()*10000
    out = out.astype('int16')
    driver = gdal.GetDriverByName("GTiff")
    utilme.save_tif2(out, None, None, save_path, driver)
    times = time.asctime(time.localtime(time.time()))
    print("#####finish time: %s######"%times)
   

if __name__ == "__main__":
    # idinit = 15
    # for i in range(idinit):
    #     print("idx %s"%i)
    #     main(i)
    main(0)

