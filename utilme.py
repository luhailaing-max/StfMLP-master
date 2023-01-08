# -*- coding: utf-8 -*-
"""
    date:20210420
    author: lhl
    function:
            show_image()：               单波段显示tif影像 20210420
            save_tif()：                 保存为tif影像 (像素大小为一个字节存储：8位。0-255)20210420
            save_tif2()：                保存为tif影像（根据原数据类型自动判断像素存储数据类型,并将保存文件名和路径作为一个参数）20210504
            hdf2tif()：                  将hdf文件转存为tif影像 20210420
            img_resize()：               单波段扩大或缩小图片尺寸（没啥用） 20210420
            log_save()：                 将logging中信息写入指定文件 20210422
            isodata_mutispectral():      对3波段光谱图像进行分类 20210425
            isodata_mutispectral6():     对6波段光谱图像进行分类 20210511
            img_resize_jpg():            针对普通图像进行重采样到指定大小 20210501
            addpad():                    针对三维图像数组，根据指定下采样分块大小，进行padding20210502
            show_imagepice()             分块显示BIGTiff中RGB遥感影像一个块 20210516
            savemodel():                 存储模型参数20210525
            loadmodel():                 加载模型参数20210525
            padding()                    对给定数组根据分块大小自动padding20210525
            getimgblock():               对给定多光谱图像，逐波段进行分块并排序，根据idx索引，取到指定分块数组20210525
            getimgblock2():              对给定多光谱图像，进行分块，每块包含所有波段，根据idx索引，取到指定分块数组20210531
            getimgblock3():对给定多光谱图像，进行分块并排序，每块包含全部波段，根据idx索引，取到指定分块数组,并增加了重叠图像块功能 20211008
            write_imgname_to_txtfile():  将指定文件夹下的所有文件名写入指定的txt文件中20210826
            write_imgname_to_txtfile1(): 将指定文件夹下的所有文件名,按四个一组（即c1,c2,f1,f2的格式）写入指定的txt文件中20210921
            load_filename_to_list():     从文本文件txt中读取每行数据，组成列表返回20210826
            test_readbil():              针对时空融合数据集CIA中的.bil文件进行统计总字节数，并读取其中某个像素点数值20210903
            test_writetotif():           针对时空融合数据集CIA中的.bil和.int文件进行读取并转存为.tif格式20210904
            readdatasist(path):          按行读取指定文件中的路径，然后按空格分割后存为数据返回20210921
            test_yaml(filepath):         使用yaml包，实现“字典”方式配置参数20211014
            test_argparse(args):         使用yaml+argparse+命名空间，实现“命名空间”方式配置参数 20211014
"""
import matplotlib.pyplot as plt
import osr
import numpy as np
import gdal
from PIL import Image
import os
import os.path
import time
import logging
import random
import math
import cv2
import torch



"""
crop_tif():

"""
def crop2tif(sdir,tdir,row,col):
    #CIA
    # top = 20
    # left = 100
    #LGC
    top = 30
    left = 30
    img = gdal.Open(sdir)
    bands = img.RasterCount
    scol = img.RasterXSize
    srow = img.RasterYSize
    image_geotrans = img.GetGeoTransform()  # 获取仿射矩阵信息
    image_projetion = img.GetProjection()  # 获取投影信息
    img_data = img.ReadAsArray()
    imgarr = img_data[:,top:row+top,left:col+left]
    bands, r, c = imgarr.shape
    datatype = gdal.GDT_UInt16
    driver = gdal.GetDriverByName("GTiff")
    datas = driver.Create(tdir, c, r, bands, datatype)

    #设置地理坐标和仿射变换信息,注意这里源图像没有添加坐标和仿射变换信息，所以继承了源图像，存储后的图像不能使用ENVI打开
    datas.SetGeoTransform(image_geotrans)
    datas.SetProjection(image_projetion)

    for i in range(bands):
        datas.GetRasterBand(i + 1).WriteArray(imgarr[i])
    del datas
    print("save successfully...")



"""
test_writetotif():针对时空融合数据集CIA中的.bil和.int文件进行读取并转存为.tif格式
    其中.bil使用read_as_bil(dataarr,bands, rows, col)读取
    .int使用read_as_bsq(dataarr,bands, rows, col)读取

"""
# 依据BIL存储规则，按照存储完一行的所有波段再存储下一行，进行提取并存入数组。
def read_as_bil(dataarr,bands, rows, col):
    imgarr = np.zeros((bands,rows,col))
    for r in range(rows): #取出一行的所有波段
        start = r * col * bands
        end = start + col * bands
        arrtem = dataarr[start:end]
        for b in range(bands): #取出每个波段
            start2 = b * col
            end2 = start2 + col
            imgarr[b,r,:] = arrtem[start2:end2]  #存入数组对应位置
    return  imgarr
# 依据BSQ存储规则，按照存储完单波段整幅图像后再存储下一波段的存储方法进行提取并存入数组。
def read_as_bsq(dataarr,bands, rows, col):
    imgarr = np.zeros((bands,rows,col))
    for b in range(bands):              #取出每个波段
        start = b * rows * col
        end = start + rows * col
        arrtem = dataarr[start:end]
        for r in range(rows):           #一维数组按行取出，存入对应三维数组。
            start2 = r * col
            end2 = start2 + col
            imgarr[b,r,:] = arrtem[start2:end2]
    return  imgarr
# 依据BIP存储规则，按照一个像素所有波段进行存储完，再存储下一个像素所有波段的存储方法进行提取并存入数组。
def read_as_bip(dataarr,bands, rows, col):
    imgarr = np.zeros((bands,rows,col))
    for r in range(rows):               #按行列遍历每个像元
        for c in range(col):
            if r == 0 :
                pix = c
            else:
                pix = r * col + c
            start = pix * bands
            end = start + bands
            arrtem = dataarr[start:end] #从一维数组中取出每个像元的全波段元素（6个）
            for b in range(bands):
                imgarr[b,r,c] = arrtem[b] # 赋值给对应数组
    return  imgarr
def test_writetotif():

    #读取二进制数据并转换成int16类型的数组
    # dir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2001_281_08oct\L71093084_08420011007_HRF_modtran_surf_ref_agd66.bil"
    dir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2001_281_08oct\MOD09GA_A2001281.sur_refl.int"

    f = open(dir, 'rb')
    fint = np.fromfile(f,dtype = np.int16)

    #数据提取
    bands, rows, col =6, 2040, 1720
    imgarr = read_as_bip(fint, bands, rows, col)

    #将提取的数组存储为tif格式图像.
    #注意这里未设置地理坐标和仿射变换信息，所以不能用ENVI等软件打开。
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2001_281_08oct\L71093084_08420011007_HRF_modtran_surf_ref_agd66.tif"
    savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2001_281_08oct\MOD09GA_A2001281.sur_refl.tif"
    datatype = gdal.GDT_UInt16
    bands, high, width = imgarr.shape
    driver = gdal.GetDriverByName("GTiff")
    datas = driver.Create(savedir, col, rows, bands, datatype)

    #设置地理坐标和仿射变换信息
    # datas.SetGeoTransform(image_geotrans)
    # datas.SetProjection(image_projetion)

    for i in range(bands):
        datas.GetRasterBand(i + 1).WriteArray(imgarr[i])
    del datas
    print("save succfully")


"""
test_readbil():
    针对时空融合数据集CIA中的.bil文件进行统计总字节数，并读取其中某个像素点数值
"""
def test_readbil():
    # dir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2001_281_08oct\L71093084_08420011007_HRF_modtran_surf_ref_agd66.bil"
    f = open(dir,'rb')
    fint = np.fromfile(f,dtype = np.int16)
    print(len(fint))    # 输出数组元素总数：21052800
    print(fint[1])      # 输出数组中第二个元素值：436
    print(fint[50000])  # 输出数组中第50001个元素值：1229

    # n = 0
    # while True:   #统计总字节数
    #     s = f.read(1)
    #     if s == b"":
    #         break
    #     n = n+1
    # print(n)   #输出：42105600


"""
    load_filename_to_list():从文本文件txt中读取每行数据，组成列表返回
        dir:文本文件txt的路径
"""
def load_filename_to_list(dir):
    img_name_list = np.loadtxt(dir, dtype=np.str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


"""
    write_imgname_to_txtfile1():将指定文件夹下的所有文件名,按六个一组（即f1,f2,f3,c1,c2,c3的格式）写入指定的txt文件中。
        imgdir:包含文件的文件夹路径。
        txtdir:需要写入的txt文件路径。 
"""
def write_imgname_to_textfile1(imgdir, txtdir):
    f = open(txtdir,'a')
    tem = []
    for file in os.listdir(imgdir):
        tem.append(file)

    tem1 = tem[0:len(tem)//2]
    tem2 = tem[len(tem)//2:len(tem)]
    for i in range(len(tem1)-2):
        f.write("{}\n".format(tem1[i] +" "+ tem1[i+1] +" " + tem1[i+2] +" "+ tem2[i] +" "+ tem2[i+1]+" "+ tem2[i+2]))
    f.close()
    print("write to txt finished...")


"""
    write_imgname_to_txtfile():将指定文件夹下的所有文件名写入指定的txt文件中。
        imgdir:包含文件的文件夹路径。
        txtdir:需要写入的txt文件路径。 
"""


def write_imgname_to_txtfile(imgdir, txtdir):
    f = open(txtdir,'a')
    for file in os.listdir(imgdir):
        f.write("{}\n".format(file))
    f.close()
    print("write to txt finished...")
#example
def example():
    imgdir = r"E:\codes\dataset\LGC\train_set"
    txtdir = r"E:\codes\codes\stf\datasets\stfganlist\LGC\totallist.txt"
    write_imgname_to_txtfile(imgdir, txtdir)
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


"""
对给定多光谱图像，逐波段进行分块并排序，每块只有一个波段，根据idx索引，取到指定分块数组
    arr: 需要执行padding操作的数组
    partrow: 每块行数
    partcol：每块列数
    idx: 序号

"""
def getimgblock(arr, idx, partrow, partcol):
    band, r, c = arr.shape
    rnum = r / partrow
    cnum = c / partcol
    idb = int(idx // (rnum * cnum))
    tem = idx % (rnum * cnum)
    idr = int(tem // cnum)
    idc = int(tem % cnum)
    idrstart = partrow * idr
    idrend = partrow * idr + partrow
    idcstart = partcol * idc
    idcend = partcol * idc + partcol

    img= arr[idb, idrstart:idrend, idcstart:idcend]
    return img


"""
对给定多光谱图像，进行分块并排序，每块包含全部波段，根据idx索引，取到指定分块数组
    arr: 需要执行padding操作的数组
    partrow: 每块行数
    partcol：每块列数
    idx: 序号

"""
def getimgblock2(arr, idx, partrow, partcol):
    band, r, c = arr.shape
    rnum = r / partrow
    cnum = c / partcol
    tem = idx
    idr = int(tem // cnum)
    idc = int(tem % cnum)
    idrstart = partrow * idr
    idrend = partrow * idr + partrow
    idcstart = partcol * idc
    idcend = partcol * idc + partcol

    # if idrend>r and idcend>c:
    #     # img = np.zeros((band,partrow,partcol))
    #     tem = arr[:,idrstart:r,idcstart:c]
    #     img=padding(tem,partcol,partcol)
    #     # img = img + tem
    #     return img
    # if idrend>r:
    #     # img = np.zeros((band,partrow,partcol))
    #     tem = arr[:,idrstart:r,idcstart:idcend]
    #     img=padding(tem,partcol,partcol)
    #
    #     # img = img + tem
    #     return img
    # if idcend>c:
    #     # img = np.zeros((band,partrow,partcol))
    #     tem = arr[:,idrstart:idrend,idcstart:c]
    #     img=padding(tem,partcol,partcol)
    #
    #     # img = img + tem
    #     return img

    img= arr[:, idrstart:idrend, idcstart:idcend]
    return img


"""
getimgblock3():对给定多光谱图像，进行分块并排序，每块包含全部波段，根据idx索引，取到指定分块数组,并增加了重叠图像块功能
    arr: 需要执行padding操作的数组
    partrow: 每块行数
    partcol：每块列数
    idx: 序号

"""
def getimgblock3(arr, idx, partrow, partcol, overlap = 0):
    band, r, c = arr.shape
    rnum = r / partrow
    cnum = c / partcol
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
readdatasist(path):按行读取指定文件中的路径，然后按空格分割后存为数据返回。
    path:数据文件的路径
    
"""
def readdatasist(path):

    listdata = []
    with open(path, 'r') as file_to_read:
        pathhead, _= os.path.split(path)
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
存储模型参数：
    m: 模型
    path: 模型存储路径，不包括文件名
    logger：记入日志
"""
def savemodel(m, path, logger = None):
    # pathhead, _ = os.path.split(path) # 将路径和文件名分开
    #path = os.path.join(pathhead, 'model_weights.pth')
    # rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    # filename = rq + "model_weights.pth"
    filename ="model_weights.pth"
    pathfile = os.path.join(path, filename)
    torch.save(m.state_dict(), pathfile)
    print("    model dict save successfully, from %s......"%pathfile)
    if logger == None:
        pass
    else:
        logger.debug("    model dict save successfully, from %s......"%pathfile)


"""
加载模型参数：
    m: 模型
    path：模型参数存储路径，包括文件名
    logger: 记入日志
"""
def loadmodel(m, path, logger = None):
    filename ="model_weights.pth"
    pathfile = os.path.join(path, filename)
    if os.path.exists(pathfile):
        m.load_state_dict(torch.load(pathfile, map_location=torch.device('cpu')))
        print("    loading model dict successfully from %s ....."%pathfile)
        if logger == None:
            pass
        else:
            logger.debug("    loading model dict successfully from %s ....."%pathfile)
    else:
        print("    loading model filed, model dict not exist, from %s....."%pathfile)
        if logger == None:
            pass
        else:
            logger.debug("    loading model filed, model dict not exist, from %s....."%pathfile)
    return m


"""
isodata多光谱分类
    dataset： 待分类数据集
    outputFilename： 分类结果输出路径
    K：初始类别数（期望）
    TN：每个类别中样本最小数目
    TS：每个类别的标准差
    TC：每个类别间的最小距离
    L： 每次允许合并的最大类别对的数量
    I：迭代次数


"""
class Pixel:
    """Pixel"""

    def __init__(self, initX: int, initY: int, initColor):
        self.x = initX
        self.y = initY
        self.color = initColor
class Cluster:
    """Cluster in Gray"""

    def __init__(self, initCenter):
        self.center = initCenter
        self.pixelList = []
class ClusterPair:
    """Cluster Pair"""

    def __init__(self, initClusterAIndex: int, initClusterBIndex: int, initDistance):
        self.clusterAIndex = initClusterAIndex
        self.clusterBIndex = initClusterBIndex
        self.distance = initDistance
def distanceBetween(colorA, colorB) -> float:

    d1 = int(colorA[0]) - int(colorB[0])
    d2 = int(colorA[1]) - int(colorB[1])
    d3 = int(colorA[2]) - int(colorB[2])
    return math.sqrt((d1 ** 2) + (d2 ** 2) + (d3 ** 2))
def distanceBetween6(colorA, colorB) -> float:

    d1 = int(colorA[0]) - int(colorB[0])
    d2 = int(colorA[1]) - int(colorB[1])
    d3 = int(colorA[2]) - int(colorB[2])
    d4 = int(colorA[3]) - int(colorB[3])
    d5 = int(colorA[4]) - int(colorB[4])
    d6 = int(colorA[5]) - int(colorB[5])
    return math.sqrt((d1 ** 2) + (d2 ** 2) + (d3 ** 2) + (d4 ** 2) + (d5 ** 2) + (d6 ** 2))
def isodata_mutispectral(dataset, outputfilename, K: int, TN: int, TS: float, TC: int, L: int, I: int):
    # dataset = gdal.Open("before.img")
    im_bands = dataset.RasterCount  # 波段数
    print("im_bands:",im_bands)
    imgX = dataset.RasterXSize  # 栅格矩阵的列数
    imgY = dataset.RasterYSize  # 栅格矩阵的行数

    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    imgArray = dataset.ReadAsArray(0, 0, imgX, imgY)  # 获取数据
    #添加以下一行，调换宽高（506行调换回来）
    imgArray = imgArray.transpose(0, 2, 1)
    print(imgArray.shape)
    print("imgx %d , imgy%d"%(imgX,imgY))
    clusterList = []
    # 随机生成聚类中心
    for i in range(0, K):

        randomX = random.randint(0, imgX - 1)
        randomY = random.randint(0, imgY- 1)

        duplicated = False

        for cluster in clusterList:
            if (cluster.center[0] == imgArray[0, randomX, randomY] and
                    cluster.center[1] == imgArray[1, randomX, randomY] and
                    cluster.center[2] == imgArray[2, randomX, randomY]
            ):
                duplicated = True
                break


        if not duplicated:
            clusterList.append(Cluster(np.array([imgArray[0, randomX, randomY],
                                                    imgArray[1, randomX, randomY],
                                                    imgArray[2, randomX, randomY]

                                                    ],
                                                   dtype=np.uint8)))

    # 开始迭代
    iterationCount = 0
    didAnythingInLastIteration = True
    while True:
        iterationCount += 1

        # 清空每一类内像元
        for cluster in clusterList:
            cluster.pixelList.clear()
        print("------")
        print("迭代第{0}次".format(iterationCount))

        # 将所有像元分类
        print("分类...", end='', flush=True)
        for row in range(0, imgX):
            for col in range(0, imgY):
                #print("row %d, col %d"%(row,col))
                targetClusterIndex = 0
                targetClusterDistance = distanceBetween(imgArray[:, row, col], clusterList[0].center)
                # 分类
                for i in range(1, len(clusterList)):
                    currentDistance = distanceBetween(imgArray[:, row, col], clusterList[i].center)
                    if currentDistance < targetClusterDistance:
                        targetClusterDistance = currentDistance
                        targetClusterIndex = i
                clusterList[targetClusterIndex].pixelList.append(Pixel(row, col, imgArray[:, row, col]))
        print(" 结束 ")

        # 检查类中样本个数是否满足要求
        gotoNextIteration = False
        for i in range(len(clusterList) - 1, -1, -1):
            if len(clusterList[i].pixelList) < TN:
                # 重新分类
                clusterList.pop(i)
                gotoNextIteration = True
                break
        if gotoNextIteration:
            print("样本个数不满足要求")
            continue
        print("样本个数满足要求")

        # 重新计算聚类中心
        print("重新计算聚类中心...", end='', flush=True)
        for cluster in clusterList:
            sum1 = 0.0
            sum2 = 0.0
            sum3 = 0.0


            for pixel in cluster.pixelList:
                sum1 += int(pixel.color[0])
                sum2 += int(pixel.color[1])
                sum3 += int(pixel.color[2])

            ave1 = round(sum1 / len(cluster.pixelList))
            ave2 = round(sum2 / len(cluster.pixelList))
            ave3 = round(sum3 / len(cluster.pixelList))


            if (ave1 != cluster.center[0] and
                    ave2 != cluster.center[1] and
                    ave3 != cluster.center[2]

            ):
                didAnythingInLastIteration = True
            cluster.center = np.array([ave1, ave2, ave3], dtype=np.uint8)

        print("结束")
        if iterationCount > I:
            break
        if not didAnythingInLastIteration:
            print("更多迭代次数是不是必要的")
            break

        # 计算平均距离
        print("准备合并或分裂...", end='', flush=True)
        aveDisctanceList = []
        sumDistanceAll = 0.0
        for cluster in clusterList:
            currentSumDistance = 0.0
            for pixel in cluster.pixelList:
                currentSumDistance += distanceBetween(pixel.color, cluster.center)
            aveDisctanceList.append(float(currentSumDistance) / len(cluster.pixelList))
            sumDistanceAll += currentSumDistance
        aveDistanceAll = float(sumDistanceAll) / (imgX * imgY)
        print(" 结束")

        if (len(clusterList) <= K / 2) or not (iterationCount % 2 == 0 or len(clusterList) >= K * 2):
            # 分裂
            print("开始分裂", end='', flush=True)
            beforeCount = len(clusterList)

            for i in range(len(clusterList) - 1, -1, -1):
                currentSD = [0.0, 0.0, 0.0]
                for pixel in clusterList[i].pixelList:
                    currentSD[0] += (int(pixel.color[0]) - int(clusterList[i].center[0])) ** 2
                    currentSD[1] += (int(pixel.color[1]) - int(clusterList[i].center[1])) ** 2
                    currentSD[2] += (int(pixel.color[2]) - int(clusterList[i].center[2])) ** 2


                currentSD[0] = math.sqrt(currentSD[0] / len(clusterList[i].pixelList))
                currentSD[1] = math.sqrt(currentSD[1] / len(clusterList[i].pixelList))
                currentSD[2] = math.sqrt(currentSD[2] / len(clusterList[i].pixelList))

                # 计算各波段最大标准差
                # Find the max in SD of R, G and B
                maxSD = currentSD[0]
                for j in (1, 2):
                    maxSD = currentSD[j] if currentSD[j] > maxSD else maxSD

                if (maxSD > TS) and (
                        (aveDisctanceList[i] > aveDistanceAll and len(clusterList[i].pixelList) > 2 * (TN + 1)) or (
                        len(clusterList) < K / 2)):
                    gamma = 0.5 * maxSD
                    clusterList[i].center[0] += gamma
                    clusterList[i].center[1] += gamma
                    clusterList[i].center[2] += gamma

                    clusterList.append(Cluster(np.array([clusterList[i].center[0],
                                                            clusterList[i].center[1],
                                                            clusterList[i].center[2]

                                                            ],
                                                           dtype=np.uint8)))
                    clusterList[i].center[0] -= gamma * 2
                    clusterList[i].center[1] -= gamma * 2
                    clusterList[i].center[2] -= gamma * 2

                    clusterList.append(Cluster(np.array([clusterList[i].center[0],
                                                            clusterList[i].center[1],
                                                            clusterList[i].center[2]

                                                            ],
                                                           dtype=np.uint8)))

                    clusterList.pop(i)
            print(" {0} -> {1}".format(beforeCount, len(clusterList)))
        elif (iterationCount % 2 == 0) or (len(clusterList) >= K * 2) or (iterationCount == I):
            # 合并
            print("合并:", end='', flush=True)
            beforeCount = len(clusterList)
            didAnythingInLastIteration = False
            clusterPairList = []
            for i in range(0, len(clusterList)):
                for j in range(0, i):
                    currentDistance = distanceBetween(clusterList[i].center, clusterList[j].center)
                    if currentDistance < TC:
                        clusterPairList.append(ClusterPair(i, j, currentDistance))

            clusterPairListSorted = sorted(clusterPairList, key=lambda clusterPair: clusterPair.distance)
            newClusterCenterList = []
            mergedClusterIndexList = []
            mergedPairCount = 0
            for clusterPair in clusterPairList:
                hasBeenMerged = False
                for index in mergedClusterIndexList:
                    if clusterPair.clusterAIndex == index or clusterPair.clusterBIndex == index:
                        hasBeenMerged = True
                        break
                if hasBeenMerged:
                    continue

                newCenter1 = int((len(clusterList[clusterPair.clusterAIndex].pixelList) * float(
                    clusterList[clusterPair.clusterAIndex].center[0]) + len(
                    clusterList[clusterPair.clusterBIndex].pixelList) * float(
                    clusterList[clusterPair.clusterBIndex].center[0])) / (
                                         len(clusterList[clusterPair.clusterAIndex].pixelList) + len(
                                     clusterList[clusterPair.clusterBIndex].pixelList)))
                newCenter2 = int((len(clusterList[clusterPair.clusterAIndex].pixelList) * float(
                    clusterList[clusterPair.clusterAIndex].center[1]) + len(
                    clusterList[clusterPair.clusterBIndex].pixelList) * float(
                    clusterList[clusterPair.clusterBIndex].center[1])) / (
                                         len(clusterList[clusterPair.clusterAIndex].pixelList) + len(
                                     clusterList[clusterPair.clusterBIndex].pixelList)))
                newCenter3 = int((len(clusterList[clusterPair.clusterAIndex].pixelList) * float(
                    clusterList[clusterPair.clusterAIndex].center[2]) + len(
                    clusterList[clusterPair.clusterBIndex].pixelList) * float(
                    clusterList[clusterPair.clusterBIndex].center[2])) / (
                                         len(clusterList[clusterPair.clusterAIndex].pixelList) + len(
                                     clusterList[clusterPair.clusterBIndex].pixelList)))


                newClusterCenterList.append(
                    [newCenter1, newCenter2, newCenter3])

                mergedClusterIndexList.append(clusterPair.clusterAIndex)
                mergedClusterIndexList.append(clusterPair.clusterBIndex)
                mergedPairCount += 1
                if mergedPairCount > L:
                    break
            if len(mergedClusterIndexList) > 0:
                didAnythingInLastIteration = True
            mergedClusterIndexListSorted = sorted(mergedClusterIndexList, key=lambda clusterIndex: clusterIndex,
                                                  reverse=True)
            for index in mergedClusterIndexListSorted:
                clusterList.pop(index)

            for center in newClusterCenterList:
                clusterList.append(Cluster(
                    np.array([center[0], center[1], center[2]],
                                dtype=np.uint8)))
            print(" {0} -> {1}".format(beforeCount, len(clusterList)))

    # 生成新的图像矩阵
    print("分类结束")
    print("一共分为 {0} 类.".format(len(clusterList)))

    newImgArray = np.zeros((3, imgX, imgY), dtype=np.uint8)
    for cluster in clusterList:
        for pixel in cluster.pixelList:
            newImgArray[0, pixel.x, pixel.y] = int(cluster.center[0])
            newImgArray[1, pixel.x, pixel.y] = int(cluster.center[1])
            newImgArray[2, pixel.x, pixel.y] = int(cluster.center[2])

    a2 = np.ones((3, imgX, imgY), dtype=np.uint8)

    unic = np.unique(newImgArray[0])
    color = []
    # print("对各个类别进行颜色渲染...")
    # for i in range(len(unic)):
    #     color.append([random.randint(0, 128), random.randint(0, 255), random.randint(128, 255)])
    #
    # for i in range(imgY):
    #     for j in range(imgX):
    #         for k in range(len(unic)):
    #             if (newImgArray[0, i, j] == unic[k]):
    #                 a2[0, i, j] = color[k][0]
    #                 a2[1, i, j] = color[k][1]
    #                 a2[2, i, j] = color[k][2]

    print("写出分类后专题图")
    driver = gdal.GetDriverByName("GTiff")
    IsoData = driver.Create(outputfilename, imgX, imgY, 3, gdal.GDT_Byte)
    # for i in range(3):
    #     IsoData.GetRasterBand(i+1).WriteArray(newImgArray[i])
    print("设置坐标参数")
    IsoData.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    print("设置投影信息")
    IsoData.SetProjection(im_proj)  # 写入投影
    #修改调换回89行调换的宽高
    newImgArray=newImgArray.transpose(0,2,1)
    for i in range(3):
        IsoData.GetRasterBand(i + 1).WriteArray(newImgArray[i])

    del dataset
    print("ISODATA非监督分类完成")
def isodata_mutispectral6(dataset, outputfilename, K: int, TN: int, TS: float, TC: float, L: int, I: int):
    # dataset = gdal.Open("before.img")
    im_bands = dataset.RasterCount  # 波段数
    print("im_bands:",im_bands)
    imgX = dataset.RasterXSize  # 栅格矩阵的列数
    imgY = dataset.RasterYSize  # 栅格矩阵的行数

    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    imgArray = dataset.ReadAsArray(0, 0, imgX, imgY)  # 获取数据
    #添加以下一行，调换宽高（506行调换回来）
    imgArray = imgArray.transpose(0, 2, 1)
    print(imgArray.shape)
    print("imgx %d , imgy%d"%(imgX,imgY))
    clusterList = []
    # 随机生成聚类中心
    for i in range(0, K):

        randomX = random.randint(0, imgX - 1)
        randomY = random.randint(0, imgY- 1)

        duplicated = False
        for cluster in clusterList:
            if (cluster.center[0] == imgArray[0, randomX, randomY] and
                    cluster.center[1] == imgArray[1, randomX, randomY] and
                    cluster.center[2] == imgArray[2, randomX, randomY] and
                    cluster.center[3] == imgArray[3, randomX, randomY] and
                    cluster.center[4] == imgArray[4, randomX, randomY] and
                    cluster.center[5] == imgArray[5, randomX, randomY]

            ):
                duplicated = True
                break

        if not duplicated:
            clusterList.append(Cluster(np.array([imgArray[0, randomX, randomY],
                                                    imgArray[1, randomX, randomY],
                                                    imgArray[2, randomX, randomY],
                                                    imgArray[3, randomX, randomY],
                                                    imgArray[4, randomX, randomY],
                                                    imgArray[5, randomX, randomY]
                                                    ],
                                                   dtype=np.uint8)))



    # 开始迭代
    iterationCount = 0
    didAnythingInLastIteration = True
    while True:
        iterationCount += 1

        # 清空每一类内像元
        for cluster in clusterList:
            cluster.pixelList.clear()
        print("------")
        print("迭代第{0}次".format(iterationCount))

        # 将所有像元分类
        print("分类...", end='', flush=True)
        for row in range(0, imgX):
            for col in range(0, imgY):
                #print("row %d, col %d"%(row,col))
                targetClusterIndex = 0
                targetClusterDistance = distanceBetween6(imgArray[:, row, col], clusterList[0].center)
                # 分类
                for i in range(1, len(clusterList)):
                    currentDistance = distanceBetween6(imgArray[:, row, col], clusterList[i].center)
                    if currentDistance < targetClusterDistance:
                        targetClusterDistance = currentDistance
                        targetClusterIndex = i
                clusterList[targetClusterIndex].pixelList.append(Pixel(row, col, imgArray[:, row, col]))
        print(" 结束 ")

        # 检查类中样本个数是否满足要求
        gotoNextIteration = False
        for i in range(len(clusterList) - 1, -1, -1):
            if len(clusterList[i].pixelList) < TN:
                # 重新分类
                clusterList.pop(i)
                gotoNextIteration = True
                break
        if gotoNextIteration:
            print("样本个数不满足要求")
            continue
        print("样本个数满足要求")

        # 重新计算聚类中心
        print("重新计算聚类中心...", end='', flush=True)
        for cluster in clusterList:
            sum1 = 0.0
            sum2 = 0.0
            sum3 = 0.0
            sum4 = 0.0
            sum5 = 0.0
            sum6 = 0.0

            for pixel in cluster.pixelList:
                sum1 += int(pixel.color[0])
                sum2 += int(pixel.color[1])
                sum3 += int(pixel.color[2])
                sum4 += int(pixel.color[3])
                sum5 += int(pixel.color[4])
                sum6 += int(pixel.color[5])
            ave1 = round(sum1 / len(cluster.pixelList))
            ave2 = round(sum2 / len(cluster.pixelList))
            ave3 = round(sum3 / len(cluster.pixelList))
            ave4 = round(sum4 / len(cluster.pixelList))
            ave5 = round(sum5 / len(cluster.pixelList))
            ave6 = round(sum6 / len(cluster.pixelList))


            if (ave1 != cluster.center[0] and
                    ave2 != cluster.center[1] and
                    ave3 != cluster.center[2] and
                    ave4 != cluster.center[3] and
                    ave5 != cluster.center[4] and
                    ave6 != cluster.center[5]
            ):
                didAnythingInLastIteration = True
            cluster.center = np.array([ave1, ave2, ave3, ave4, ave5, ave6, ave6], dtype=np.uint8)


        print("结束")
        if iterationCount > I:
            break
        if not didAnythingInLastIteration:
            print("更多迭代次数是不是必要的")
            break

        # 计算平均距离
        print("准备合并或分裂...", end='', flush=True)
        aveDisctanceList = []
        sumDistanceAll = 0.0
        for cluster in clusterList:
            currentSumDistance = 0.0
            for pixel in cluster.pixelList:
                currentSumDistance += distanceBetween6(pixel.color, cluster.center)
            aveDisctanceList.append(float(currentSumDistance) / len(cluster.pixelList))
            sumDistanceAll += currentSumDistance
        aveDistanceAll = float(sumDistanceAll) / (imgX * imgY)
        print(" 结束")

        if (len(clusterList) <= K / 2) or not (iterationCount % 2 == 0 or len(clusterList) >= K * 2):
            # 分裂
            print("开始分裂", end='', flush=True)
            beforeCount = len(clusterList)
            for i in range(len(clusterList) - 1, -1, -1):
                currentSD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                for pixel in clusterList[i].pixelList:
                    currentSD[0] += (int(pixel.color[0]) - int(clusterList[i].center[0])) ** 2
                    currentSD[1] += (int(pixel.color[1]) - int(clusterList[i].center[1])) ** 2
                    currentSD[2] += (int(pixel.color[2]) - int(clusterList[i].center[2])) ** 2
                    currentSD[3] += (int(pixel.color[3]) - int(clusterList[i].center[3])) ** 2
                    currentSD[4] += (int(pixel.color[4]) - int(clusterList[i].center[4])) ** 2
                    currentSD[5] += (int(pixel.color[5]) - int(clusterList[i].center[5])) ** 2

                currentSD[0] = math.sqrt(currentSD[0] / len(clusterList[i].pixelList))
                currentSD[1] = math.sqrt(currentSD[1] / len(clusterList[i].pixelList))
                currentSD[2] = math.sqrt(currentSD[2] / len(clusterList[i].pixelList))
                currentSD[3] = math.sqrt(currentSD[3] / len(clusterList[i].pixelList))
                currentSD[4] = math.sqrt(currentSD[4] / len(clusterList[i].pixelList))
                currentSD[5] = math.sqrt(currentSD[5] / len(clusterList[i].pixelList))


                # 计算各波段最大标准差
                # Find the max in SD of R, G and B
                maxSD = currentSD[0]
                for j in (1, 2):
                    maxSD = currentSD[j] if currentSD[j] > maxSD else maxSD

                if (maxSD > TS) and (
                        (aveDisctanceList[i] > aveDistanceAll and len(clusterList[i].pixelList) > 2 * (TN + 1)) or (
                        len(clusterList) < K / 2)):
                    gamma = 0.5 * maxSD
                    clusterList[i].center[0] += gamma
                    clusterList[i].center[1] += gamma
                    clusterList[i].center[2] += gamma
                    clusterList[i].center[3] += gamma
                    clusterList[i].center[4] += gamma
                    clusterList[i].center[5] += gamma


                    clusterList.append(Cluster(np.array([clusterList[i].center[0],
                                                            clusterList[i].center[1],
                                                            clusterList[i].center[2],
                                                            clusterList[i].center[3],
                                                            clusterList[i].center[4],
                                                            clusterList[i].center[5]
                                                            ],
                                                           dtype=np.uint8)))
                    clusterList[i].center[0] -= gamma * 2
                    clusterList[i].center[1] -= gamma * 2
                    clusterList[i].center[2] -= gamma * 2
                    clusterList[i].center[3] -= gamma * 2
                    clusterList[i].center[4] -= gamma * 2
                    clusterList[i].center[5] -= gamma * 2

                    clusterList.append(Cluster(np.array([clusterList[i].center[0],
                                                            clusterList[i].center[1],
                                                            clusterList[i].center[2],
                                                            clusterList[i].center[3],
                                                            clusterList[i].center[4],
                                                            clusterList[i].center[5]

                                                            ],
                                                           dtype=np.uint8)))



                    clusterList.pop(i)
            print(" {0} -> {1}".format(beforeCount, len(clusterList)))
        elif (iterationCount % 2 == 0) or (len(clusterList) >= K * 2) or (iterationCount == I):
            # 合并
            print("合并:", end='', flush=True)
            beforeCount = len(clusterList)
            didAnythingInLastIteration = False
            clusterPairList = []
            for i in range(0, len(clusterList)):
                for j in range(0, i):
                    currentDistance = distanceBetween6(clusterList[i].center, clusterList[j].center)
                    if currentDistance < TC:
                        clusterPairList.append(ClusterPair(i, j, currentDistance))

            clusterPairListSorted = sorted(clusterPairList, key=lambda clusterPair: clusterPair.distance)
            newClusterCenterList = []
            mergedClusterIndexList = []
            mergedPairCount = 0
            for clusterPair in clusterPairList:
                hasBeenMerged = False
                for index in mergedClusterIndexList:
                    if clusterPair.clusterAIndex == index or clusterPair.clusterBIndex == index:
                        hasBeenMerged = True
                        break
                if hasBeenMerged:
                    continue

                newCenter1 = int((len(clusterList[clusterPair.clusterAIndex].pixelList) * float(
                    clusterList[clusterPair.clusterAIndex].center[0]) + len(
                    clusterList[clusterPair.clusterBIndex].pixelList) * float(
                    clusterList[clusterPair.clusterBIndex].center[0])) / (
                                             len(clusterList[clusterPair.clusterAIndex].pixelList) + len(
                                         clusterList[clusterPair.clusterBIndex].pixelList)))
                newCenter2 = int((len(clusterList[clusterPair.clusterAIndex].pixelList) * float(
                    clusterList[clusterPair.clusterAIndex].center[1]) + len(
                    clusterList[clusterPair.clusterBIndex].pixelList) * float(
                    clusterList[clusterPair.clusterBIndex].center[1])) / (
                                             len(clusterList[clusterPair.clusterAIndex].pixelList) + len(
                                         clusterList[clusterPair.clusterBIndex].pixelList)))
                newCenter3 = int((len(clusterList[clusterPair.clusterAIndex].pixelList) * float(
                    clusterList[clusterPair.clusterAIndex].center[2]) + len(
                    clusterList[clusterPair.clusterBIndex].pixelList) * float(
                    clusterList[clusterPair.clusterBIndex].center[2])) / (
                                             len(clusterList[clusterPair.clusterAIndex].pixelList) + len(
                                         clusterList[clusterPair.clusterBIndex].pixelList)))
                newCenter4 = int((len(clusterList[clusterPair.clusterAIndex].pixelList) * float(
                    clusterList[clusterPair.clusterAIndex].center[3]) + len(
                    clusterList[clusterPair.clusterBIndex].pixelList) * float(
                    clusterList[clusterPair.clusterBIndex].center[3])) / (
                                             len(clusterList[clusterPair.clusterAIndex].pixelList) + len(
                                         clusterList[clusterPair.clusterBIndex].pixelList)))
                newCenter5 = int((len(clusterList[clusterPair.clusterAIndex].pixelList) * float(
                    clusterList[clusterPair.clusterAIndex].center[4]) + len(
                    clusterList[clusterPair.clusterBIndex].pixelList) * float(
                    clusterList[clusterPair.clusterBIndex].center[4])) / (
                                             len(clusterList[clusterPair.clusterAIndex].pixelList) + len(
                                         clusterList[clusterPair.clusterBIndex].pixelList)))
                newCenter6 = int((len(clusterList[clusterPair.clusterAIndex].pixelList) * float(
                    clusterList[clusterPair.clusterAIndex].center[5]) + len(
                    clusterList[clusterPair.clusterBIndex].pixelList) * float(
                    clusterList[clusterPair.clusterBIndex].center[5])) / (
                                             len(clusterList[clusterPair.clusterAIndex].pixelList) + len(
                                         clusterList[clusterPair.clusterBIndex].pixelList)))


                newClusterCenterList.append(
                    [newCenter1, newCenter2, newCenter3, newCenter4, newCenter5, newCenter6])

                mergedClusterIndexList.append(clusterPair.clusterAIndex)
                mergedClusterIndexList.append(clusterPair.clusterBIndex)
                mergedPairCount += 1
                if mergedPairCount > L:
                    break
            if len(mergedClusterIndexList) > 0:
                didAnythingInLastIteration = True
            mergedClusterIndexListSorted = sorted(mergedClusterIndexList, key=lambda clusterIndex: clusterIndex,
                                                  reverse=True)
            for index in mergedClusterIndexListSorted:
                clusterList.pop(index)


            for center in newClusterCenterList:
                clusterList.append(Cluster(
                    np.array([center[0], center[1], center[2], center[3], center[4], center[5]],
                                dtype=np.uint8)))

            print(" {0} -> {1}".format(beforeCount, len(clusterList)))

    # 生成新的图像矩阵
    print("分类结束")
    print("一共分为 {0} 类.".format(len(clusterList)))

    newImgArray = np.zeros((6, imgX, imgY), dtype=np.uint8)
    for cluster in clusterList:
        for pixel in cluster.pixelList:
            newImgArray[0, pixel.x, pixel.y] = int(cluster.center[0])
            newImgArray[1, pixel.x, pixel.y] = int(cluster.center[1])
            newImgArray[2, pixel.x, pixel.y] = int(cluster.center[2])
            newImgArray[3, pixel.x, pixel.y] = int(cluster.center[3])
            newImgArray[4, pixel.x, pixel.y] = int(cluster.center[4])
            newImgArray[5, pixel.x, pixel.y] = int(cluster.center[5])

    # a2 = np.ones((3, imgX, imgY), dtype=np.uint8)
    #
    # unic = np.unique(newImgArray[0])
    # color = []
    # print("对各个类别进行颜色渲染...")
    # for i in range(len(unic)):
    #     color.append([random.randint(0, 128), random.randint(0, 255), random.randint(128, 255)])
    #
    # for i in range(imgY):
    #     for j in range(imgX):
    #         for k in range(len(unic)):
    #             if (newImgArray[0, i, j] == unic[k]):
    #                 a2[0, i, j] = color[k][0]
    #                 a2[1, i, j] = color[k][1]
    #                 a2[2, i, j] = color[k][2]

    print("写出分类后专题图")
    driver = gdal.GetDriverByName("GTiff")
    IsoData = driver.Create(outputfilename, imgX, imgY, newImgArray.shape[0], gdal.GDT_Byte)
    # for i in range(3):
    #     IsoData.GetRasterBand(i+1).WriteArray(newImgArray[i])
    print("设置坐标参数")
    IsoData.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    print("设置投影信息")
    IsoData.SetProjection(im_proj)  # 写入投影
    #修改调换回89行调换的宽高
    newImgArray=newImgArray.transpose(0,2,1)
    for i in range(newImgArray.shape[0]):
        IsoData.GetRasterBand(i + 1).WriteArray(newImgArray[i])

    del dataset
    print("ISODATA非监督分类完成")


"""
使用logging包实现打印日志到终端屏幕同时输出日志到指定文件夹。
    path:日志文件存放路径（例：r"E:\codes\stf\logs"）
    logfilename:日志文件名字（例："test.log")
    20210422
"""
def log_save(path,logfilename):
    os.chdir(path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)  # 设置日志格式
    # # 第一步，输出到屏幕
    # sh = logging.StreamHandler()  # 往屏幕上输出
    # sh.setFormatter(format_str)  # 设置屏幕上显示的格式
    # logger.addHandler(sh)

    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    fh = logging.FileHandler(logfilename, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    fh.setFormatter(format_str)
    logger.addHandler(fh)

    return logger


"""
按波段显示图像，单波段图像可以不指定波段值
    imgpath: 图像路径（例：r"E:\codes\stf\dataset\AHB\M_2018-5-12.tif"）
    band: 波段值 从0开始
"""
def show_image(imgpath, band=999):
    img = gdal.Open(imgpath)
    bands = img.RasterCount
    img_width = img.RasterXSize
    img_height = img.RasterYSize
    imgmata = img.GetMetadataItem("PHOTOMETRIC", "DMD_CREATIONOPTIONLIST")
    # print("img metadata: ", imgmata)
    # img_gtf = img.GetGeoTransform()  # 获取仿射矩阵信息
    # img_proj = img.GetProjection()  # 获取投影信息
    # print("bands:",bands)
    print("img_width:",img_width)
    print("img_height:",img_height)
    img_data = img.ReadAsArray()
    # print(type(img_data))
    # img_data = img_data.astype('int16')
    # # sum = np.sum(img_data)
    # img_data = img_data - 255

    # print("image_data:",img_data)
    if len(img_data.shape) < 2:
        print("this image just has one band ")
        # print("img_data:", img_data)
        plt.figure('landsat: img_data')
        plt.imshow(img_data)
        plt.show()
        return
    if band == 999:
        print("please enter a band number! example:show_image_band(img_x,3).")
        return
    if band >= bands:
        print("out range of bands, it should be < ", bands)
        return
    print("show image in band " + str(band))
    print("img:", img.GetProjection())

    img1_band = img_data[band, 0:img_height, 0:img_width]
    print("imag_band:", img1_band)
    # for i in range(img_width):
    #     for j in range(img_height):
    #         if img1_band[i][j]==0:
    #             print(img1_band[i][j],end=' ')
    plt.figure('landsat: img_peer_band')
    plt.imshow(img1_band)
    plt.show()


"""
分块显示BIGTiff中RGB遥感影像一个块
    imgpath: 文件路径
    band: 指定一个波段
    rows: 显示从0行开始指定的行数
    cols：显示从0列开始的指定列数
"""
def show_imagepice(imgpath, band, rows, cols):
    img = gdal.Open(imgpath)
    bands = img.RasterCount
    img_width = img.RasterXSize
    img_height = img.RasterYSize
    imgmata = img.GetMetadataItem("PHOTOMETRIC", "DMD_CREATIONOPTIONLIST")
    # print("img metadata: ", imgmata)
    # img_gtf = img.GetGeoTransform()  # 获取仿射矩阵信息
    # img_proj = img.GetProjection()  # 获取投影信息

    print("bands:",bands)
    print("img_width:",img_width)
    print("img_height:",img_height)
    if band >= bands:
        print("out range of bands, it should be < ", bands)
        return
    img_datas = img.ReadAsArray(100000, 100000, rows, cols)
    print(img_datas)
    plt.figure('landsat: img_peer_band')
    plt.imshow(img_datas[0])
    plt.show()


"""
保存单波段或者多个波段合为一副tif影像
    image_array: 从图像读取道德数组
    image_geotrans: tif仿射变换信息（在text()中有获取方法）
    image_projetion: tif图像投影信息
    filename: 保存tif图像的名称（例：a.tif）
    save_path: 保存图像的路径（例：r"E:\codes\stf\dataset\AHB"）
"""
def save_tif(image_array, image_geotrans, image_projetion, filename, save_path,driver):
    os.chdir(save_path)
    # if'int8' in image_array[0].dtype.name:
    #     datatype = gdal.GDT_Byte
    # elif "int16" in image_array[0].dtype.name:
    #     datatype = gdal.GDT_UInt16
    # else:
    #     datatype = gdal.GDT_UInt32
    datatype = gdal.GDT_Byte

    if len(image_array.shape) >= 3:
        bands, high, width = image_array.shape
    else:
        bands, (high, width) = 1, image_array.shape

    datas = driver.Create(filename, width, high, bands, datatype)
    datas.SetGeoTransform(image_geotrans)
    datas.SetProjection(image_projetion)
    if bands == 1:
        datas.GetRasterBand(1).WriteArray(image_array)
    else:
        for i in range(bands):
            datas.GetRasterBand(i+1).WriteArray(image_array[i])
    del datas
    print("save succfully")
    return
def save_tif2(image_array, image_geotrans, image_projetion, save_path, driver):
    # os.chdir(save_path)
    if'int8' in image_array[0].dtype.name:
        datatype = gdal.GDT_Byte
    elif "int16" in image_array[0].dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_UInt32
    # datatype = gdal.GDT_Byte

    if len(image_array.shape) >= 3:
        bands, high, width = image_array.shape
    else:
        bands, (high, width) = 1, image_array.shape

    datas = driver.Create(save_path, width, high, bands, datatype)
    if image_geotrans is not None:
        datas.SetGeoTransform(image_geotrans)
    if image_projetion is not None:
        datas.SetProjection(image_projetion)
    if bands == 1:
        datas.GetRasterBand(1).WriteArray(image_array)
    else:
        for i in range(bands):
            datas.GetRasterBand(i+1).WriteArray(image_array[i])
    del datas
    print("save succfully")
    return


"""
读取hdf图像，并保存指定波段为tif
hdf2tif():
    hdfpath: hdf文件存储路径（例：r"E:\codes\stf\dataset\hdf"）
    
array2raster():
    tifname：保存tif图像的名称（例：a.tif）
    GeoTransform：仿射变换信息
    array：需要保存的图像数组，必须是二维
    save_path: 保存图像的路径（例：r"E:\codes\stf\dataset\hdf"）
"""
# hdf单波段保存tif#  hdf批量转tif
def array2raster(tifname, GeoTransform, array, save_path):
    os.chdir(save_path)
    cols = array.shape[1]  # 矩阵列数
    rows = array.shape[0]  # 矩阵行数
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(tifname, cols, rows, 1, gdal.GDT_Float32)
    # 括号中两个0表示起始像元的行列号从(0,0)开始
    outRaster.SetGeoTransform(tuple(GeoTransform))
    # 获取数据集第一个波段，是从1开始，不是从0开始
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    # 代码4326表示WGS84坐标
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
def hdf2tif(hdfpath):
    #  获取文件夹内的文件名
    hdfNameList = os.listdir(hdfpath)
    save_path=hdfpath
    for i in range(len(hdfNameList)):
        #  判断当前文件是否为HDF文件
        if (os.path.splitext(hdfNameList[i])[1] == ".hdf"):
            hdfPath = hdfpath + "/" + hdfNameList[i]
            #  gdal打开hdf数据集
            datasets = gdal.Open(hdfPath)
            #  获取hdf中的元数据
            Metadata = datasets.GetMetadata()
            print("Metadata:", Metadata)
            #  获取四个角的维度
            Latitudes = Metadata["GRINGPOINTLATITUDE.1"]
            #  采用", "进行分割
            LatitudesList = Latitudes.split(", ")
            #  获取四个角的经度
            Longitude = Metadata["GRINGPOINTLONGITUDE.1"]
            #  采用", "进行分割
            LongitudeList = Longitude.split(", ")
            # 图像四个角的地理坐标
            GeoCoordinates = np.zeros((4, 2), dtype="float32")
            GeoCoordinates[0] = np.array([float(LongitudeList[0]), float(LatitudesList[0])])
            GeoCoordinates[1] = np.array([float(LongitudeList[1]), float(LatitudesList[1])])
            GeoCoordinates[2] = np.array([float(LongitudeList[2]), float(LatitudesList[2])])
            GeoCoordinates[3] = np.array([float(LongitudeList[3]), float(LatitudesList[3])])
            #  列数
            Columns = float(Metadata["DATACOLUMNS500M"])
            #  行数
            Rows = float(Metadata["DATAROWS500M"])
            #  图像四个角的图像坐标
            PixelCoordinates = np.array([[0, 0],
                                         [Columns - 1, 0],
                                         [Columns - 1, Rows - 1],
                                         [0, Rows - 1]], dtype="float32")
            #  计算仿射变换矩阵
            from scipy.optimize import leastsq
            def func(i):
                Transform0, Transform1, Transform2, Transform3, Transform4, Transform5 = i[0], i[1], i[2], i[3], i[4], \
                                                                                         i[5]
                return [Transform0 + PixelCoordinates[0][0] * Transform1 + PixelCoordinates[0][1] * Transform2 -
                        GeoCoordinates[0][0],
                        Transform3 + PixelCoordinates[0][0] * Transform4 + PixelCoordinates[0][1] * Transform5 -
                        GeoCoordinates[0][1],
                        Transform0 + PixelCoordinates[1][0] * Transform1 + PixelCoordinates[1][1] * Transform2 -
                        GeoCoordinates[1][0],
                        Transform3 + PixelCoordinates[1][0] * Transform4 + PixelCoordinates[1][1] * Transform5 -
                        GeoCoordinates[1][1],
                        Transform0 + PixelCoordinates[2][0] * Transform1 + PixelCoordinates[2][1] * Transform2 -
                        GeoCoordinates[2][0],
                        Transform3 + PixelCoordinates[2][0] * Transform4 + PixelCoordinates[2][1] * Transform5 -
                        GeoCoordinates[2][1],
                        Transform0 + PixelCoordinates[3][0] * Transform1 + PixelCoordinates[3][1] * Transform2 -
                        GeoCoordinates[3][0],
                        Transform3 + PixelCoordinates[3][0] * Transform4 + PixelCoordinates[3][1] * Transform5 -
                        GeoCoordinates[3][1]]

            #  最小二乘法求解
            GeoTransform = leastsq(func, np.asarray((1, 1, 1, 1, 1, 1)))
            #  获取数据时间
            date = Metadata["RANGEBEGINNINGDATE"]
            # 获取指定波段数据
            datasetsshape = datasets.GetSubDatasets()
            for i in range(len(datasetsshape)):
                print("datasetsshape content,%d----%s:" % (i, datasetsshape[i]))
            #hdf文件中含有多个图像信息，包括1km和500米，数组第一维从11开始后的6个波段为500米波段
            Dataset = datasets.GetSubDatasets()[12][0]
            Raster = gdal.Open(Dataset)
            arr = Raster.ReadAsArray()
            print("arr:", arr)
            TifName = date + ".tif"
            array2raster(TifName, GeoTransform[0], arr, save_path)
            print(TifName, "Saved successfully!")


"""
#图像单波段重采样到指定分辨率
    srcarr：原图像数组
    xsize：目标图像宽
    ysize：目标图像高
"""
def img_resize(srcarr, xsize, ysize):
    srcimg = Image.fromarray(srcarr)
    dstarr = np.array(srcimg.resize((xsize, ysize)))
    return dstarr


"""
# jpg图像缩放 20210501
    src: 源图像路径（例：r"E:\origin.jpg"）
    dst: 缩放后的图像存储路径，需带文件名（例：r"E:\1.jpg"）
"""
def img_resize_jpg(src, dst, xsize, ysize):
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    print("img", img.shape)
    img = cv2.resize(img, (xsize, ysize))
    print("img.resize", img.shape)
    cv2.imwrite(dst, img)


"""
# 三维数组分块不足，扩展补0，输入下采样每个像素对应的像素行列数，通过计算，得出需要额外padding的行列数
    arr: 需要padding的目标数组
    row: 影像分块行像素个数{例：要把影像降维：16*16像素作为粗像素的一个像素，则row为16}
    col: 影像分块列像素个数

"""
def addpad(arr, row, col):
    padrownum = row-arr.shape[1] % row
    padcolnum = col-arr.shape[2] % col
    print("数据扩充，末尾增加%d行，%d列。"%(padrownum, padcolnum))
    return np.pad(arr, ((0, 0), (0, padrownum), (0, padcolnum)), "constant")




"""
    test_yaml(filepath):使用yaml包，实现“字典”方式配置参数
        filepath:配置文件路径
"""
import yaml
def test_yaml(filepath):
    with open(filepath, 'r') as f: # 用with读取文件更好
        configs = yaml.load(f, Loader=yaml.FullLoader) # 按字典格式读取并返回
    # 显示读取后的内容
    # print(type(configs)) #<class 'dict'>
    # print(configs["stage1"]) #{'number': 3, 'banchsize': 32}
    # print(configs["stage1"]['number']) # 3
    # print(configs['model']) # {'level1': {'name': 'cnn', 'layernumber': 4}, 'level2': {'name': 'lstm', 'layernumber': 12}, 'level3': {'name': 'resnet50', 'layernumber': 50, 'layernormal': True}}
    # print(configs['model']["level3"]['layernumber'])# 50
    # print(configs['loggfile'])# loggle.log
    return configs


"""
    test_argparse(args):使用yaml+argparse+命名空间，实现“命名空间”方式配置参数 20211014
        args: 是argparse实例
    
    example:
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg',type = str,default=r"E:\codes\codes\stf\a.yaml",help="...") # a.yaml中内容在文章开始给出
        args = parser.parse_args()
        test_argparse(args)

"""
import yaml
import argparse
def dict2namespace(config):
    #声明命名空间
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        #将参数对添加到命名空间中
        setattr(namespace, key, new_value)
    return namespace

def test_argparse(args):

    filepath = args.cfg

    with open(filepath, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    configs = dict2namespace({**config, **vars(args)})
    # 测试输出
    # print(type(configs))                    #<class 'argparse.Namespace'>
    # print(configs.model)                    #Namespace(level1=Namespace(layernumber=4, name='cnn'), level2=Namespace(layernumber=12, name='lstm'), level3=Namespace(layernormal=True, layernumber=50, name='resnet50'))
    # print(configs.stage1)                   #Namespace(banchsize=32, number=3)
    # print(configs.stage1.number)            #3
    # print(configs.model)                    #Namespace(level1=Namespace(layernumber=4, name='cnn'), level2=Namespace(layernumber=12, name='lstm'), level3=Namespace(layernormal=True, layernumber=50, name='resnet50'))
    # print(configs.model.level3.layernumber) #50
    # print(configs.loggfile)                 #loggle.log

    return configs



def test1():


    # # spath1 =  r"E:\codes\dataset\biascnn\biastest\AHB\L_2018-5-12.tif"
    # x = r"E:\datasets\stfdatasets\LGC\LGC\Landsat\2004_107_Apr16\20040416_TM.tif"
    #
    # # show_image(spath1,5)
    # show_image(x, 3)
    # show_image(x, 4)
    # show_image(x, 5)



    # sdir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2001_313_09nov\L71093084_08420011108_HRF_modtran_surf_ref_agd66.tif"

    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2001_281_08oct\L71093084_08420011007_HRF_modtran_surf_ref_agd66.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2001_290_17oct\L71093084_08420011016_HRF_modtran_surf_ref_agd66.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2001_306_02nov\L71093084_08420011101_HRF_modtran_surf_ref_agd66.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2001_313_09nov\L71093084_08420011108_HRF_modtran_surf_ref_agd66.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2001_329_25nov\L71093084_08420011124_HRF_modtran_surf_ref_agd66.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2001_338_04dec\L71093084_08420011203_HRF_modtran_surf_ref_agd66.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2002_005_05jan\L71093084_08420020104_HRF_modtran_surf_ref_agd66.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2002_012_12jan\L71093084_08420020111_HRF_modtran_surf_ref_agd66.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2002_044_13feb\L71093084_08420020212_HRF_modtran_surf_ref_agd66.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2002_053_22feb\L71093084_08420020221_HRF_modtran_surf_ref_agd66.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2002_069_10mar\L72093084_08420020309_HRF_modtran_surf_ref_agd66.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2002_076_17mar\L71093084_08420020316_HRF_modtran_surf_ref_agd66.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2002_092_02apr\L71093084_08420020401_HRF_modtran_surf_ref_agd66.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2002_101_11apr\L71093084_08420020410_HRF_modtran_surf_ref_agd66.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2002_108_18apr\L71093084_08420020417_HRF_modtran_surf_ref_agd66.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2002_117_27apr\L71093084_08420020426_HRF_modtran_surf_ref_agd66.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2002_124_04may\L71093084_08420020503_HRF_modtran_surf_ref_agd66.tif"

    # tdir = r"E:\codes\dataset\CIA\L2001_281_08oct.tif"
    # tdir = r"E:\codes\dataset\CIA\L2001_290_17oct.tif"
    # tdir = r"E:\codes\dataset\CIA\L2001_306_02nov.tif"
    # tdir = r"E:\codes\dataset\CIA\L2001_313_09nov.tif"
    # tdir = r"E:\codes\dataset\CIA\L2001_329_25nov.tif"
    # tdir = r"E:\codes\dataset\CIA\L2001_338_04dec.tif"
    # tdir = r"E:\codes\dataset\CIA\L2002_005_05jan.tif"
    # tdir = r"E:\codes\dataset\CIA\L2002_012_12jan.tif"
    # tdir = r"E:\codes\dataset\CIA\L2002_044_13feb.tif"
    # tdir = r"E:\codes\dataset\CIA\L2002_053_22feb.tif"
    # tdir = r"E:\codes\dataset\CIA\L2002_069_10mar.tif"
    # tdir = r"E:\codes\dataset\CIA\L2002_076_17mar.tif"
    # tdir = r"E:\codes\dataset\CIA\L2002_092_02apr.tif"
    # tdir = r"E:\codes\dataset\CIA\L2002_101_11apr.tif"
    # tdir = r"E:\codes\dataset\CIA\L2002_108_18apr.tif"
    # tdir = r"E:\codes\dataset\CIA\L2002_117_27apr.tif"
    # tdir = r"E:\codes\dataset\CIA\L2002_124_04may.tif"

    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2001_281_08oct\MOD09GA_A2001281.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2001_290_17oct\MOD09GA_A2001290.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2001_306_02nov\MOD09GA_A2001306.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2001_313_09nov\MOD09GA_A2001313.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2001_329_25nov\MOD09GA_A2001329.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2001_338_04dec\MOD09GA_A2001338.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2002_005_05jan\MOD09GA_A2002005.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2002_012_12jan\MOD09GA_A2002012.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2002_044_13feb\MOD09GA_A2002044.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2002_053_22feb\MOD09GA_A2002053.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2002_069_10mar\MOD09GA_A2002069.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2002_076_17mar\MOD09GA_A2002076.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2002_092_02apr\MOD09GA_A2002092.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2002_101_11apr\MOD09GA_A2002101.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2002_108_18apr\MOD09GA_A2002108.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2002_117_27apr\MOD09GA_A2002117.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2002_124_04may\MOD09GA_A2002124.sur_refl.tif"

    # tdir = r"E:\codes\dataset\CIA\M2001_281_08oct.tif"
    # tdir = r"E:\codes\dataset\CIA\M2001_290_17oct.tif"
    # tdir = r"E:\codes\dataset\CIA\M2001_306_02nov.tif"
    # tdir = r"E:\codes\dataset\CIA\M2001_313_09nov.tif"
    # tdir = r"E:\codes\dataset\CIA\M2001_329_25nov.tif"
    # tdir = r"E:\codes\dataset\CIA\M2001_338_04dec.tif"
    # tdir = r"E:\codes\dataset\CIA\M2002_005_05jan.tif"
    # tdir = r"E:\codes\dataset\CIA\M2002_012_12jan.tif"
    # tdir = r"E:\codes\dataset\CIA\M2002_044_13feb.tif"
    # tdir = r"E:\codes\dataset\CIA\M2002_053_22feb.tif"
    # tdir = r"E:\codes\dataset\CIA\M2002_069_10mar.tif"
    # tdir = r"E:\codes\dataset\CIA\M2002_076_17mar.tif"
    # tdir = r"E:\codes\dataset\CIA\M2002_092_02apr.tif"
    # tdir = r"E:\codes\dataset\CIA\M2002_101_11apr.tif"
    # tdir = r"E:\codes\dataset\CIA\M2002_108_18apr.tif"
    # tdir = r"E:\codes\dataset\CIA\M2002_117_27apr.tif"
    # tdir = r"E:\codes\dataset\CIA\M2002_124_04may.tif"



    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\Landsat\2004_107_Apr16\20040416_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\L20040416_TM.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\Landsat\2004_123_May02\20040502_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\L20040502_TM.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\Landsat\2004_187_Jul05\20040705_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\L20040705_TM.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\Landsat\2004_219_Aug06\20040806_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\L20040806_TM.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\Landsat\2004_235_Aug22\20040822_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\L20040822_TM.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\Landsat\2004_299_Oct25\20041025_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\L20041025_TM.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\Landsat\2004_331_Nov26\20041126_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\L20041126_TM.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\Landsat\2004_347_Dec12\20041212_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\L20041212_TM.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\Landsat\2004_363_Dec28\20041228_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\L20041228_TM.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\Landsat\2005_013_Jan13\20050113_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\L20050113_TM.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\Landsat\2005_029_Jan29\20050129_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\L20050129_TM.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\Landsat\2005_045_Feb14\20050214_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\L20050214_TM.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\Landsat\2005_061_Mar02\20050302_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\L20050302_TM.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\Landsat\2005_093_Apr03\20050403_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\L20050403_TM.tif"

    #
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\MODIS\2004_107_Apr16\MOD09GA_A2004107.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\MODIS\2004_123_May02\MOD09GA_A2004123.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\MODIS\2004_187_Jul05\MOD09GA_A2004187.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\MODIS\2004_219_Aug06\MOD09GA_A2004219.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\MODIS\2004_235_Aug22\MOD09GA_A2004235.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\MODIS\2004_299_Oct25\MOD09GA_A2004299.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\MODIS\2004_331_Nov26\MOD09GA_A2004331.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\MODIS\2004_347_Dec12\MOD09GA_A2004347.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\MODIS\2004_363_Dec28\MOD09GA_A2004363.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\MODIS\2005_013_Jan13\MOD09GA_A2005013.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\MODIS\2005_029_Jan29\MOD09GA_A2005029.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\MODIS\2005_045_Feb14\MOD09GA_A2005045.sur_refl.tif"
    # savedir = r"E:\datasets\stfdatasets\LGC\LGC\MODIS\2005_061_Mar02\MOD09GA_A2005061.sur_refl.tif"
    savedir = r"E:\datasets\stfdatasets\LGC\LGC\MODIS\2005_093_Apr03\MOD09GA_A2005093.sur_refl.tif"
    #
    # tdir = r"E:\codes\dataset\LGC\M20040416_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\M20040502_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\M20040705_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\M20040806_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\M20040822_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\M20041025_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\M20041126_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\M20041212_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\M20041228_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\M20050113_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\M20050129_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\M20050214_TM.tif"
    # tdir = r"E:\codes\dataset\LGC\M20050302_TM.tif"
    tdir = r"E:\codes\dataset\LGC\M20050403_TM.tif"

    # rows = 1792
    # cols = 1280



    # rows = 2560
    # cols = 3072
    #
    # crop2tif(savedir,tdir,rows,cols)

    x =  r"E:\codes\dataset\LGC\MODIS\M20050403_TM.tif"

    # show_image(spath1,5)
    show_image(x, 3)
    show_image(x, 4)
    show_image(x, 5)



def test2():
    # xpath = r"E:\datasets\stfdatasets\CIA\CIA\MODIS\2001_281_08oct\MOD09GA_A2001281.sur_refl.tif"
    # # ypath = r"E:\codes\dataset\CIA\train_set\L2001_290_17oct.tif"
    # show_image(xpath, 0)
    # # show_image(ypath, 0)


    # inputFilename = r'E:\codes\dataset\CIA\train_set\L2002_053_22feb.tif'
    # outputFilename = r"E:\codes\dataset\CIA\STARFM-FSDAF-test_results\isodata-classfy-L2002_053_22feb.tif"

    inputFilename = r'E:\codes\dataset\CIA\train_set\L2002_101_11apr.tif'
    outputFilename = r"E:\codes\dataset\CIA\STARFM-FSDAF-test_results\isodata-classfy-L2002_101_11apr.tif"

    argvK = 8  # 初始类别数（期望）
    argvTN = 20  # 每个类别中样本最小数目
    argvTS = 1  # 每个类别的标准差
    argvTC = 0.5  # 每个类别间的最小距离
    argvL = 5  # 每次允许合并的最大类别对的数量
    argvI = 10  # 迭代次数
    dataset = gdal.Open(inputFilename)
    isodata_mutispectral6(dataset, outputFilename, argvK, argvTN, argvTS, argvTC, argvL, argvI)

    return

def test():
    imgdir = r"E:\codes\dataset\2021prcv_contest\test_set\image1"
    txtdir = r"E:\codes\dataset\2021prcv_contest\test_set\list\val.txt"
    write_imgname_to_txtfile(imgdir, txtdir)

def test111():
    a = torch.randn((4,4))
    print(a)
    b = a[0:2,0:2]
    print(b)
    c = a[2:4,2:5]
    print(c)

def test3():
    # filepath = os.path.join(os.getcwd(), 'a.yaml') # 文件路径,这里需要将a.yaml文件与本程序文件放在同级目录下
    # test_yaml(filepath)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=r"E:\codes\codes\stf\a.yaml", help="...")  # a.yaml中内容在文章开始给出
    args = parser.parse_args()
    test_argparse(args)

def show_img():
    x = r"E:\codes\codes\stf\baseline\stfnet\logs\v1\stfganlist\L2001_313_09nov_v1_100000_stfganlist.tif"
    show_image(x,0)
    targetbefor = r"E:\codes\dataset\CIA\train_set\L2001_306_02nov.tif"
    show_image(targetbefor,0)
    target = r"E:\codes\dataset\CIA\train_set\L2001_313_09nov.tif"
    show_image(target,0)
    targetafter = r"E:\codes\dataset\CIA\train_set\L2001_329_25nov.tif "
    show_image(targetafter,0)
if __name__ == "__main__":
    # example()

    # imgpath = r"E:\stfnew_L20040502_TM.tif"
    imgpath = r"D:\codes\dataset\LGC\train_set\L20040705_TM.tif"
    show_image(imgpath, 0)

# 查看BIGTiff图像中的某一块可视化
"""
    imgpath = r"E:\codes\datasets\WHUBuilding\WHUBuilding.tif"
    show_imagepice(imgpath, 3, 555, 5550)
    # show_image(imgpath, 0)
    return
"""
# tif_show显示指定波段图像,单波段图像可以不指定波段值
"""
    # imgpath = r"E:\codes\stf\dataset\origin\isodata-classfy-L72000306_SZ_B432_30m.tif"

    # imgpath = r"E:\codes\stf\dataset\AHB\isodata-classfy-L_2018-3-25.tif"


    imgpath = r"E:\codes\stf\dataset\AHB\l2_pre_fsdaf.tif"
    imgpath1 = r"E:\codes\stf\dataset\AHB\L_2018-5-12.tif"

    show_image(imgpath, 0)
    show_image(imgpath1, 0)
    return
"""

# img_resize: 多波段影像下采样+padding+保存为tif影像
"""
    imgpath = r"E:\MOD09_2002311_SZ_B214_250m.tif"
    img = gdal.Open(imgpath)
    transform = img.GetGeoTransform()
    projection = img.GetProjection()
    driver = gdal.GetDriverByName("GTiff")
    img_data = img.ReadAsArray()
    img_data = addpad(img_data, 16, 16)
    if len(img_data.shape) >= 3:
        bands, high, width = img_data.shape
    else:
        bands, high, width = 1, img_data.shape
    imglow = []

    for i in range(bands):
        print("src %d shape is :%s" % (i, img_data[i].shape))
        dst = img_resize(img_data[i], 32, 32)
        imglow.append(dst)
        print("dst %d shape is :%s" % (i, dst.shape))
    imglow = np.array(imglow)
    dstpath = "E://"
    filename = "low_MOD09_2002311_SZ_B214_250m.tif"
    save_tif2(imglow,transform, projection, filename, dstpath, driver)

    return
"""

# 测试isodata分类
"""
    # inputFilename = r"E:\codes\stf\dataset\origin\L72000306_SZ_B432_30m.tif"
    # outputFilename = r"E:\codes\stf\dataset\origin\isodata-classfy-L72000306_SZ_B432_30m.tif"
    inputFilename = r"E:\codes\stf\dataset\AHB\L_2018-3-25.tif"
    outputFilename = r"E:\codes\stf\dataset\AHB\isodata-classfy-L_2018-3-25.tif"

    argvK = 8  # 初始类别数（期望）
    argvTN = 20  # 每个类别中样本最小数目
    argvTS = 1  # 每个类别的标准差
    argvTC = 0.5  # 每个类别间的最小距离
    argvL = 5  # 每次允许合并的最大类别对的数量
    argvI = 10  # 迭代次数
    dataset = gdal.Open(inputFilename)
    isodata_mutispectral(dataset, outputFilename, argvK, argvTN, argvTS, argvTC, argvL, argvI)

    return
"""

# 测试logging包，将log信息写入文件
"""
    a="aaaabbbbcccc"
    b=666
    path= r"E:\codes\stf\logs"
    logger = log_save(path, "test.log")
    logger.info('bulabula')
    logger.debug(a)
"""
# hdf格式存为tif
"""   
    hdfdir = r"E:\codes\stf\dataset\hdf"
    hdf2tif(hdfdir)
"""

# img_resize: 图像单波段重采样
"""   
    imgpath = r"E:\codes\stf\dataset\AHB\M_2018-5-12.tif"
    img = gdal.Open(imgpath)
    img_data = img.ReadAsArray()
    if len(img_data.shape) >= 3:
        bands, high, width = img_data.shape
    else:
        bands, high, width = 1, img_data.shape
    for i in range(bands):
        print("src %d shape is :%s" % (i, img_data[i].shape))
        dst = img_resize(img_data[i], width // 30, high // 30)
        print("dst %d shape is :%s" % (i, dst.shape))
    return
"""

# img_resize_jpg: 普通图像缩放
"""
    src = r"E:\origin.jpg"
    dst = r"E:\1.jpg"
    img_resize_jpg(src,dst)

    return
"""

#save_tif图像的读取和存储
"""   
    filepath = r"E:\codes\stf\dataset\AHB\L_2018-5-12.tif"
    save_path = r"E:\codes\stf\dataset\AHB"
    image = gdal.Open(filepath)
    image_array = image.ReadAsArray()/10000.*255
    print("image_array shape:", image_array.shape)
    transform = image.GetGeoTransform()
    projection = image.GetProjection()


    driver = gdal.GetDriverByName("GTiff")
    save_tif(image_array, transform, projection, "test.tif", save_path,driver)
"""

# save_tif2图像的读取和存储
"""
    filepath = r"E:\codes\stf\dataset\AHB\L_2018-5-12.tif"
    save_path = r"E:\codes\stf\dataset\AHB\HI.tif"
    image = gdal.Open(filepath)
    image_array = image.ReadAsArray() / 10000. * 255
    print("image_array shape:", image_array.shape)
    transform = image.GetGeoTransform()
    projection = image.GetProjection()

    driver = gdal.GetDriverByName("GTiff")
    save_tif2(image_array, transform, projection, save_path, driver)
    return
"""
