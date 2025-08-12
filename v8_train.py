# -*- coding: utf-8 -*-
"""

"""
from skimage import io
import os
import torch
from torch.utils.data import Dataset
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch
import os
import random
from torch.nn import functional as F
from torch import  nn, einsum
#
import torch
import os
import numpy as np
import time
import random

import argparse
import torch.optim as optim
from torch import  nn, einsum

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR

"""
readdatasist(path):按行读取指定文件中的路径，然后按空格分割后存为数据返回。
    path:数据文件的路径

"""
def readdatasist(path):
    listdata = []
    with open(path, 'r') as file_to_read:
        pathhead, _ = os.path.split(path)
        print(pathhead)
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            if not lines:
                break
                # pass
            teml = [i for i in lines.split()]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            listdata.append(teml)  # 添加新读取的数据
            # pass
    # listdata = np.array(listdata)  # 将数据从list类型转换为array类型。
    # pass
    return listdata
"""
getimgblock3():对给定多光谱图像，进行分块并排序，每块包含全部波段，根据idx索引，取到指定分块数组,并增加了重叠图像块功能
    arr: 需要执行padding操作的数组
    partrow: 每块行数
    partcol：每块列数
    idx: 序号

"""
import math
def getimgblock3(arr, idx, partrow, partcol, overlap = 0):
    band, r, c = arr.shape
    # rnum = r / partrow
    # cnum = c / partcol
    rnum = math.ceil(r / partrow)
    cnum = math.ceil(c / partcol)
    tem = idx
    idr = int(tem // cnum)
    idc = int(tem % cnum)
    idrstart = partrow * idr
    idrend = partrow * idr + partrow
    idcstart = partcol * idc
    idcend = partcol * idc + partcol

    if (idrstart - overlap) >= 0:
        idrstart-=overlap

    idrend+= overlap
    if (idcstart-overlap) >= 0:
        idcstart -=overlap

    idcend += overlap
    img= arr[:, idrstart:idrend, idcstart:idcend]
    return img
"""
padding，根据快行数和列数自动计算需要paddingd的行数和列数
    arr: 需要执行padding操作的数组
    partrow: 每块行数
    partcol：每块列数
"""
def padding(arr, partrow, partcol):
    band, r, c = arr.shape
    # print("padding before %s"%str(arr.shape))
    if r % partrow == 0:
        row = r
    else:
        row = r + (partrow - r % partrow)
    if c % partcol == 0:
        col = c
    else:
        col = c + (partcol - c % partcol)
    rowp = row - r
    colp = col - c
    arr = np.pad(arr, ((0, 0), (0, rowp), (0, colp)), "constant")
    # print("padding after %s"%str(arr.shape))
    return arr

class MyDataset(Dataset):
    def __init__(self, img_path, img_list, patchrow, patchcol, overlap = 0):
        self.img_list = img_list
        self.img_path = img_path
        self.patchrow = patchrow
        self.patchcol = patchcol
        self.overlap = overlap

        dirf1 = os.path.join(self.img_path,self.img_list[0])
        dirf2 = os.path.join(self.img_path,self.img_list[1])
        dirf3 = os.path.join(self.img_path,self.img_list[2])
        dirc1 = os.path.join(self.img_path,self.img_list[3])
        dirc2 = os.path.join(self.img_path,self.img_list[4])
        dirc3 = os.path.join(self.img_path,self.img_list[5])

        # c1 = gdal.Open(dirc1).ReadAsArray()
        # c2 = gdal.Open(dirc2).ReadAsArray()
        # c3 = gdal.Open(dirc3).ReadAsArray()
        # f1 = gdal.Open(dirf1).ReadAsArray()
        # f2 = gdal.Open(dirf2).ReadAsArray()
        # f3 = gdal.Open(dirf3).ReadAsArray()

        c1 = io.imread(dirc1).transpose((2,0,1))
        c2 = io.imread(dirc2).transpose((2,0,1))
        c3 = io.imread(dirc3).transpose((2,0,1))
        f1 = io.imread(dirf1).transpose((2,0,1))
        f2 = io.imread(dirf2).transpose((2,0,1))
        f3 = io.imread(dirf3).transpose((2,0,1))


        self.band, self.r,self.c = c1.shape

        c1 = c1.astype('int16')
        c2 = c2.astype('int16')
        c3 = c3.astype('int16')
        f1 = f1.astype('int16')
        f2 = f2.astype('int16')
        f3 = f3.astype('int16')

        c1 = torch.tensor(c1)
        c2 = torch.tensor(c2)
        c3 = torch.tensor(c3)
        f1 = torch.tensor(f1)
        f2 = torch.tensor(f2)
        f3 = torch.tensor(f3)

        # c12 = torch.abs(c2/10000-c1/10000)
        # c23 = torch.abs(c3/10000-c2/10000)
        # sumc12 = torch.sum(c12)
        # sumc23=torch.sum(c23)
        # self.v12 = sumc12/(self.band*self.r*self.c)
        # self.v23 = sumc23/(self.band*self.r*self.c)

        c12 = c2/10000-c1/10000
        c23 = c3/10000-c2/10000
        self.v12 =torch.abs( torch.sum(c12)/(self.band*self.r*self.c))
        self.v23=torch.abs(torch.sum(c23)/(self.band*self.r*self.c))



        self.c1 = padding(c1,patchrow,patchcol)
        self.c2 = padding(c2,patchrow,patchcol)
        self.c3 = padding(c3,patchrow,patchcol)

        self.f1 = padding(f1,patchrow,patchcol)
        self.f2 = padding(f2,patchrow,patchcol)
        self.f3 = padding(f3,patchrow,patchcol)

        _,self.paddr,self.paddc = self.c1.shape

    def __len__(self):
        # path = os.path.join(self.img_path,self.img_list[0])
        # img0 = gdal.Open(path).ReadAsArray()
        band, r, c = self.c1.shape
        rnum = r / self.patchrow
        cnum = c / self.patchcol
        num = int(rnum * cnum)
        # print("b,%s r,%s c %s"%(band,r,c))
        # print("total num is ",num)
        return num

    def __getitem__(self, idx):
        # print("idx:%s"%idx)
        c1 = getimgblock3(self.c1,idx,self.patchrow,self.patchcol,self.overlap)
        c2 = getimgblock3(self.c2,idx,self.patchrow,self.patchcol,self.overlap)
        c3 = getimgblock3(self.c3,idx,self.patchrow,self.patchcol,self.overlap)
        f1 = getimgblock3(self.f1,idx,self.patchrow,self.patchcol,self.overlap)
        f2 = getimgblock3(self.f2,idx,self.patchrow,self.patchcol,self.overlap)
        f3 = getimgblock3(self.f3,idx,self.patchrow,self.patchcol,self.overlap)

        # c1 = c1.astype('int16')
        # c2 = c2.astype('int16')
        # c3 = c3.astype('int16')
        # f1 = f1.astype('int16')
        # f2 = f2.astype('int16')
        # f3 = f3.astype('int16')
        #
        # c1 = torch.tensor(c1)
        # c2 = torch.tensor(c2)
        # c3 = torch.tensor(c3)
        # f1 = torch.tensor(f1)
        # f2 = torch.tensor(f2)
        # f3 = torch.tensor(f3)

        output = {
            "idx":idx,
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "c":self.c,
            "r": self.r,
            "paddc": self.paddc,
            "paddr": self.paddr,
            "v12":self.v12,
            "v23":self.v23
                  }
        return output

# -*- coding: utf-8 -*-



#
BN_MOMENTUM = 0.01
# BN_CHANNEL = 60

class Mlp(nn.Module):
    def __init__(self, in_features,out_features=None, mlpratio = 2,act_layer=nn.GELU, drop=0.):
        super().__init__()

        self.fc1 = nn.Linear(in_features, in_features*mlpratio)
        self.act = act_layer()
        self.fc2 = nn.Linear(in_features*mlpratio, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Basicblock(nn.Module):
    def __init__(self,indim, outdim):
        super(Basicblock, self).__init__()

        self.layernorm = nn.LayerNorm(indim)
        self.mlp = Mlp(indim, outdim )

    def forward(self, x):
        x = self.layernorm(x)
        x = self.mlp(x)
        return x


class Fist_layer(nn.Module):

    def __init__(self, in_chans=3, out_chans=96):
        super().__init__()
        self.proj = Mlp(in_chans,out_chans)
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

    def flops(self):

        return None


class Last_layer(nn.Module):

    def __init__(self, in_chans=96, out_chans=6):
        super().__init__()

        self.proj = Mlp(in_chans, out_chans)
        self.outchannel = out_chans
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2).view(B, self.outchannel, H, W)
        return x
    def flops(self):

        return None

class transition(nn.Module):
    def __init__(self, in_chans=96, out_chans=96):
        super().__init__()
        self.proj = Mlp(in_chans*in_chans, out_chans*out_chans)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x


class Mlpnet(nn.Module):
    def __init__(self, cfg= None):
        super(Mlpnet, self).__init__()
        self.fuse = cfg.fuse
        self.dim = cfg.STAGE1.DIM[0]
        self.inchannel = cfg.inchannel
        self.outchannel = cfg.outchannel
        self.layernum = cfg.STAGE1.NUM_BLOCKS[0]
        self.fistlayer = Fist_layer(self.inchannel,self.dim)
        #stage1
        self.stage1 = self._make_layer()
        # #stage2
        self.stage2_cfg = cfg.STAGE2
        self.transition1 = transition(self.stage2_cfg.RESOLUTION[0],self.stage2_cfg.RESOLUTION[1])
        self.stage2 = self._make_stage(self.stage2_cfg)

        #stage3
        self.stage3_cfg = cfg.STAGE3
        self.transition2 = transition(self.stage3_cfg.RESOLUTION[1],self.stage3_cfg.RESOLUTION[2])
        self.stage3 = self._make_stage(self.stage3_cfg)

        #stage4
        self.stage4_cfg = cfg.STAGE4
        self.transition3 = transition(self.stage4_cfg.RESOLUTION[2],self.stage4_cfg.RESOLUTION[3])
        self.stage4 = self._make_stage(self.stage4_cfg)

        self.final_dim =  np.int(np.sum(self.stage4_cfg.DIM))


        self.lastlayer = Last_layer(self.final_dim,self.outchannel)

    def _make_layer(self):
        layers = nn.ModuleList()
        for i in range(self.layernum):
            layer = Basicblock(self.dim,self.dim)
            layers.append(layer)
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config):
        num_branches = layer_config.NUM_BRANCHES
        num_blocks = layer_config.NUM_BLOCKS
        dim = layer_config.DIM
        resolution = layer_config.RESOLUTION
        modules = []
        for i in range(num_branches):
            modules.append(self._make_one_branch(i, num_blocks,dim))
        return nn.Sequential(*modules)

    def _make_one_branch(self, branch_index, num_blocks, dim):
        layers = []
        for i in range(num_blocks[branch_index]):
            layers.append(Basicblock(dim[branch_index],dim[branch_index]))
        return nn.Sequential(*layers)


    def forward(self,x):
        _,_,w,h = x.shape
        x = self.fistlayer(x)

        # stage1
        x = self.stage1(x)

        # stage2
        x_list = []
        for i in range(self.stage2_cfg.NUM_BRANCHES):
            if (i+1) == self.stage2_cfg.NUM_BRANCHES:
                x_list.append(self.transition1(x))
            else:
                x_list.append(x)
        y_list = []
        for i in range(2):
            y_list.append(self.stage2[i](x_list[i]))


        #stage3
        x_list = []
        for i in range(self.stage3_cfg.NUM_BRANCHES):
            if (i+1) == self.stage3_cfg.NUM_BRANCHES:
                x_list.append(self.transition2(y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = []
        for i in range(3):
            y_list.append(self.stage3[i](x_list[i]))


        #stage4
        x_list = []
        for i in range(self.stage4_cfg.NUM_BRANCHES):
            if (i+1) == self.stage4_cfg.NUM_BRANCHES:
                x_list.append(self.transition3(y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = []
        for i in range(4):
            y_list.append(self.stage4[i](x_list[i]))

        x_list = []
        for i in range(4):
            B,N,C = y_list[i].shape
            SIZE = int(np.sqrt(N))
            x_list.append(y_list[i].transpose(1, 2).view(B, C, SIZE, SIZE))

        x0 = x_list[0]
        x1 = F.upsample(x_list[1], size=(h, w), mode='bilinear')
        x2 = F.upsample(x_list[2], size=(h, w), mode='bilinear')
        x3 = F.upsample(x_list[3], size=(h, w), mode='bilinear')
        x  = torch.cat((x0,x1,x2,x3),dim=1)

        x = self.lastlayer(x)
        return x

def get_model(cfg, **kwargs):
    model = Mlpnet(cfg)
    return  model


class StfNet(nn.Module):
    def __init__(self,config):
        super(StfNet, self).__init__()

        self.n1 = get_model(config)
        self.n2 = get_model(config)

    def forward(self,c1,c2,c3,f1,f3):
        c1c2diff = c2 - c1
        c2c3diff= c3 - c2
        c1c3diff = c3 - c1
        x1 = self.n1( torch.cat([c1c3diff,f1],dim=1))
        x3 = self.n1( torch.cat([c1c2diff,f1],dim=1))

        x2 = self.n2( torch.cat([c1c3diff,f3],dim=1))
        x4 = self.n2( torch.cat([c2c3diff,f3],dim=1))
        return x1,x2,x3,x4




class Loss(nn.Module):
    def __init__(self,mybate,lamb=0.5):
        super(Loss, self).__init__()
        self.lamb = lamb
        # self.mybate = mybate
    def forward(self,x1,x2,x3,x4,f1,f3):
        f1f3diff = f3 - f1
        #论文公式9
        l1 = F.mse_loss(x1,f1f3diff)
        l2 = F.mse_loss(x2,f1f3diff)
        LR = l1+l2
        #论文公式10，公式11
        tem = x3+x4
        # tem= self.mybate*x3+(1-self.mybate)*x4
        # tem= self.mybate*x4+(1-self.mybate)*x3
        l3 = F.mse_loss(tem,f1f3diff)
        # 论文公式8
        loss =LR+self.lamb*l3
        # loss = (1. - self.lamb) * LR + self.lamb * l3
        return loss


import yaml
import argparse

def dict2namespace(config):
    # 声明命名空间
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        # 将参数对添加到命名空间中
        setattr(namespace, key, new_value)
    return namespace

def test_argparse(args,configpath):
    filepath = configpath
    with open(filepath, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    configs = dict2namespace({**config, **vars(args)})
    return configs

# configpath ="/kaggle/input/config/v2_config.yaml"
# # parser = argparse.ArgumentParser()
# # parser.add_argument('--cfg', type=str, default=configpath, help="...")  #
# # args = parser.parse_args()
# args = argparse.Namespace()
# configs = test_argparse(args,configpath)

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

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''

        torch.save(model.state_dict(), self.path)
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saved model successfully ...')

        self.val_loss_min = val_loss



def main(index):

    # configpath = "/kaggle/input/configs/v7_config.yaml"
    # configpath =r"D:\codes\codes\stf\mlp\v7\v7_config.yaml"
    # configpath ="/home/hllu/codes/stf/mlp/v7/v7_config.yaml"

    # configpath = "/kaggle/input/configs/v8_config.yaml"
    # configpath =r"D:\codes\codes\stf\mlp\v8\v8_train_config.yaml"
    configpath ="/home/hllu/codes/stf/mlp/v8/v8_train_config.yaml"

    args = argparse.Namespace()
    configs = test_argparse(args,configpath)
    logpath = configs.logpath
    # #
    print("logpath is : %s"%logpath)
    # txtfile = configs.txtfile
    # checkpoints_name =configs.checkpoints_name
    # checkpoint_path = os.path.join(configs.logpath, checkpoints_name)
    checkpoint_path = configs.checkpoint_path
    version = configs.version
#     idx=configs.idx+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    idx=index
    batch_size = configs.batch_size
    patchrow = configs.img_size
    patchcol = configs.img_size
    print("idx is :",idx)


#ADAMW
    epochs = configs.epochs
    seed = configs.seed
    log_freq = configs.log_freq
    savemodel_frequence = configs.savemodel_frequence
    eps = configs.eps
    betas = (configs.betas1,configs.betas2)
    base_lr = configs.base_lr
    min_lr = configs.min_lr
    warmup_lr = configs.warmup_lr
    weight_decay = configs.weight_decay
    warmup_epochs = configs.warmup_epochs
# SGD
#   seed = 1234
    sgdlr = 0.1
    momentem = 0.9
    step_size = 300
    gamma = 0.1


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed_everything(seed)

    checkpoints_name= configs.checkpoints_name
    # imglistdir = os.path.join(configs.listdir,txtfile)
    imglistdir = configs.imglistdir
    datalist = readdatasist(imglistdir)

    model = StfNet(configs).to(device)
    
    print(model)
    if os.path.exists(checkpoint_path):
        # model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))

        print("    Success to loading model dict from %s ....."%checkpoint_path)
        check_epoch = checkpoints_name.split('.')[0]
        check_epoch = int(check_epoch.split('_')[-1])
    else:
        print("    Failed to load model dict  from %s ....."%checkpoint_path)
        check_epoch = 0


    # print(model)
    if torch.cuda.device_count() > 1:
        # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        # model = torch.nn.DataParallel(model, device_ids=[1])
        model = torch.nn.DataParallel(model)

    traindata = MyDataset(configs.trainpath, datalist[idx], patchrow, patchcol)
    train_loader = DataLoader(dataset=traindata, batch_size=batch_size, shuffle=True)

    # temdata = traindata.__getitem__(0)
    # v12 = temdata["v12"]
    # v23 = temdata["v23"]
    # mybate = v23/(v12+v23)
    # criterion = Loss(mybate=mybate,lamb=0.5)

    criterion = Loss(mybate=0.0,lamb=1.0)
#ADAMW
    # optimizer = optim.AdamW(model.parameters(), eps=eps, betas=betas,
    #                         lr=base_lr, weight_decay=weight_decay)
    # n_iter_per_epoch = len(train_loader)
    # num_steps = int((epochs - check_epoch) * n_iter_per_epoch)
    # warmup_steps = int(warmup_epochs * n_iter_per_epoch)
    #
    #
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=20)
#SGD
    optimizer = optim.SGD(model.parameters(), lr=sgdlr,momentum=momentem,weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    # scheduler = MultiStepLR(optimizer, [50, 80, 120,150], gamma)

    writer= SummaryWriter(logpath)
    epochs = 600
    savemodel_frequence = 300
    # initialize the early_stopping object
    pathname = version + "_idx_" + str(idx) + "_" + ".pth"
    # 在本地服务器的存储文件夹存储模型
    bestdicpath=os.path.join(logpath, pathname)
    early_stopping = EarlyStopping(patience=600, verbose=True, path=bestdicpath)



    for epoch in range(check_epoch,epochs):
        model.train()
        totalloss = 0.
        for i, data in enumerate(train_loader):
            c1 = data["c1"].to(device)
            c2 = data["c2"].to(device)
            c3 = data["c3"].to(device)
            f1 = data["f1"].to(device)
            f3 = data["f3"].to(device)
            # label = data["f2"].to(device)

            c1 = c1.to(torch.float32)/10000
            c2 = c2.to(torch.float32)/10000
            c3 = c3.to(torch.float32)/10000

            f1 = f1.to(torch.float32)/10000
            f3 = f3.to(torch.float32) / 10000
            # label = label.to(torch.float32)/10000
            x1, x2, x3, x4 = model(c1, c2, c3, f1, f3)
            # if channels == 6:
            #     output = model(c1,c2,f1)
            # elif bandstart == 0:
            #     output = model(c1[:,0:3,:,:],c2[:,0:3,:,:],f1[:,0:3,:,:])
            #     label= label[:, 0:3, :, :]
            # else:
            #     output = model(c1[:, 3:, :, :], c2[:, 3:, :, :], f1[:, 3:, :, :])
            #     label= label[:, 3:, :, :]
            loss = criterion(x1,x2,x3,x4,f1,f3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # lr_scheduler.step_update(epoch * num_steps + (i+len(train_loader)))
            # lr_scheduler.step()
#             if i % log_freq == 0:
#                 times = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
#                 print("%s:epoch %s/%s--step:%s/%s--loss:%s" % (
#                     times, epoch, epochs, i, len(train_loader), str(loss.item())))
#                 writer.add_scalar('loss', loss.item(), global_step=i)
            totalloss += loss.item()
        times = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        print("%s:epoch %s/%s-avgtra-loss ***:%s" % (times, epoch, epochs, str(totalloss / (len(train_loader)))))
        writer.add_scalar('avg-trainloss'+str(idx), totalloss / (len(train_loader)), global_step=epoch)
        # print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler.step()

        # if (epoch + 1) % savemodel_frequence == 0:
        #     pathname = version+"_idx_" +str(idx)+"_"+ str(epoch + 1) + ".pth"
        #     #在本地服务器的存储文件夹存储模型
        #     torch.save(model.state_dict(), os.path.join(logpath, pathname))
        #     #在kaggle上的存储文件夹存储模型
        #     # torch.save(model.state_dict(), ("./"+pathname))
        #     print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        #     times = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        #     print("%s:save %s successfully" % (times, pathname))

        ave_trainloss = totalloss / (len(train_loader))
        early_stopping(ave_trainloss, model)

        if early_stopping.early_stop:
            print("Early stopping on epoch:%d....."%(epoch+1))
            break

if __name__ == "__main__":
    index=15
    for i in range(index):
        # print(i)
        main(i)
    # index=4
    # main(index)



