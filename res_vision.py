import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
import torch
import gdal
import numpy
import os
import cv2 as cv



def extend_0_to_255(img):
    img = numpy.array(img)
    max = numpy.max(img)
    min = numpy.min(img)
    b, r, c = img.shape
    imgcd = img
    for i in range(b):
        for j in range(r):
            for k in range(c):
                imgcd[i][j][k] = 255 * (img[i][j][k] - min) / (max - min)
    # print(max, min)
    return imgcd
def saveaspng():
    #cutted
    # im = gdal.Open(r"E:\codes\dataset\CIA\train_set\M2001_338_04dec.tif").ReadAsArray() #
    # im = gdal.Open(r"E:\codes\dataset\CIA\train_set\L2001_338_04dec.tif").ReadAsArray() #

    # im = gdal.Open(r"E:\codes\dataset\LGC\train_set\L20050129_TM.tif").ReadAsArray() #
    # im = gdal.Open(r"E:\codes\dataset\LGC\train_set\M20050129_TM.tif").ReadAsArray() #

    #original
    # im = gdal.Open(r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\Landsat\2001_338_04dec\L71093084_08420011203_HRF_modtran_surf_ref_agd66.tif").ReadAsArray() #
    # im = gdal.Open(r"E:\datasets\stfdatasets\Coleambally_Irrigation_Area\CIA\MODIS\2001_338_04dec\MOD09GA_A2001338.sur_refl.tif").ReadAsArray() #

    # im = gdal.Open(r"E:\datasets\stfdatasets\LGC\LGC\Landsat\2005_029_Jan29\20050129_TM.tif").ReadAsArray() #
    # im = gdal.Open(r"E:\datasets\stfdatasets\LGC\LGC\MODIS\2005_029_Jan29\MOD09GA_A2005029.sur_refl.tif").ReadAsArray() #

    #experiencement result
    # im = gdal.Open(r"E:\codes\dataset\res\stfmlp\CIA\stfmlp_L2001_338_04dec.tif").ReadAsArray() #
    im = gdal.Open(r"F:\Ediskbackup\datasets\stfdatasets\LGC\LGC\MODIS\2004_107_Apr16\MOD09GA_A2004107.sur_refl.tif").ReadAsArray() #
    # im = gdal.Open( r"E:\stfnew_L20040502_TM.tif").ReadAsArray()

    target = r"E:\codes\dataset\res\resvision"
    # cia 和 LGC需要除以10000， 李军老师的三个数据集不需要。
    im = im/10000
    im = torch.tensor(im[ 3: : ])
    img = transforms.ToPILImage()(im).convert('RGB')
    plt.figure('landsat: img_data')

    img = extend_0_to_255(img)

    #cutted
    # Image.fromarray(imgcd.astype("uint8")).convert("RGB").save(os.path.join(target,"ciamodis338.png"))
    # Image.fromarray(img.astype("uint8")).convert("RGB").save(os.path.join(target,"cialandsat338.png"))

    # Image.fromarray(img.astype("uint8")).convert("RGB").save(os.path.join(target,"lgcL20050129_TM.png"))
    # Image.fromarray(img.astype("uint8")).convert("RGB").save(os.path.join(target,"lgcM20050129_TM.png"))

    #original
    # Image.fromarray(img.astype("uint8")).convert("RGB").save(os.path.join(target,"ciaoriginal338L71093084_08420011203_HRF_modtran_surf_ref_agd66.png"))
    # Image.fromarray(img.astype("uint8")).convert("RGB").save(os.path.join(target,"ciaoriginal338MOD09GA_A2001338.sur_refl.png"))

    # Image.fromarray(img.astype("uint8")).convert("RGB").save(os.path.join(target,"lgcoriginalL20050129_TM.png"))
    # Image.fromarray(img.astype("uint8")).convert("RGB").save(os.path.join(target,"lgcoriginalMOD09GA_A2005029.sur_refl.png"))

    #experiencement result
    # Image.fromarray(img.astype("uint8")).convert("RGB").save(os.path.join(target,"cia-stfmlp_L2001_338_04dec.png"))
    # Image.fromarray(img.astype("uint8")).convert("RGB").save(os.path.join(target,"AHB_Landsat.png"))

    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def enlarge():

    #CIA LGC dst
    #CIA
    # img_path = r"D:\codes\dataset\res\resvision\original\cutted"
    # imgname = r"cialandsat338.png"
    # #LGC
    # imgname = r"lgcL20050129_TM.png"
    #
    # target = r"D:\codes\dataset\res\resvision"
    # dir = os.path.join(img_path, imgname)
    # imgarr = Image.open(dir)
    # dst = numpy.array(imgarr)
    # #CIA
    # # cv.rectangle(dst, (250, 300), (550,600), (255, 255, 0), 5)
    # #LGC
    # cv.rectangle(dst, (550, 1200), (850,1500), (255, 255, 0), 5)
    #
    # plt.imshow(dst)
    # plt.show()
    # Image.fromarray(dst.astype("uint8")).convert("RGB").save(os.path.join(target,imgname))



    # CIA LGC enlarge
    img_path = r"D:\codes\dataset\res\resvision\original\cutted"
    #CIA
    # imgname = r"cialandsat338.png"
    # LGC
    imgname = r"lgcL20050129_TM.png"

    target = r"D:\codes\dataset\res\resvision\enlarged"
    dir = os.path.join(img_path, imgname)
    imgarr = Image.open(dir)
    dst = numpy.array(imgarr)
    #CIA
    # part = dst[300:600, 250:550]
    #LGC
    part = dst[1200:1500, 550:850]
    mask = cv.resize(part, (600, 600), fx=0, fy=0, interpolation=cv.INTER_LINEAR)
    plt.imshow(mask)
    plt.show()
    Image.fromarray(part.astype("uint8")).convert("RGB").save(os.path.join(target,imgname))


import numpy as np
from skimage import io
def v2processimage(dir, target, tname, mode, saved = False):
    imgarr = io.imread(dir)
    # print(imgarr)
    output = imgarr[:, :, ]

    output = np.float32(output)
    dst = np.zeros(output.shape, dtype=np.float32)
    cv.normalize(output, dst=dst, alpha=0, beta=1.0, norm_type=cv.NORM_MINMAX)

    dst = np.uint8(dst * 255)
    io.imshow(dst)
    # print(dst)
    if mode == "cia":
        #cia
        part = dst[400:500, 350:450]
        mask = cv.resize(part, (500, 500), fx=0, fy=0, interpolation=cv.INTER_LINEAR)
        # 图像大小：1280*1800
        # 右上坐标
        dst[50:550, 730:1230] = mask
        cv.rectangle(dst, (350, 400), (450, 500), (255, 0, 0), 12)
        cv.rectangle(dst, (730, 50), (1230, 550), (255, 0, 0), 12)

        dst = cv.line(dst, (450, 500), (730, 550), (255, 0, 0), 12)
        dst = cv.line(dst, (450, 400), (730, 50), (255, 0, 0), 12)
    else:
        # lgc
        # part = dst[1300:1400, 600:700]
        # mask = cv.resize(part, (500, 500), fx=0, fy=0, interpolation=cv.INTER_LINEAR)
        # # 右下坐标
        # dst[1250:1750, 730:1230] = mask
        # cv.rectangle(dst, (700, 1400), (600, 1300), (255, 0, 0), 8)
        # cv.rectangle(dst, (730, 1250), (1230, 1750), (255, 0, 0), 8)
        #
        # dst = cv.line(dst, (700, 1300), (730, 1250), (255, 0, 0), 8)
        # dst = cv.line(dst, (700, 1400), (730, 1750), (255, 0, 0), 8)

        part = dst[1300:1400, 600:700]
        mask = cv.resize(part, (500, 500), fx=0, fy=0, interpolation=cv.INTER_LINEAR)
        # 右下坐标
        # dst[600:1100, 650:1150] = mask
        # cv.rectangle(dst, (700, 1400), (600, 1300), (255, 0, 0), 8)
        # cv.rectangle(dst, (650, 600), (1150, 1100), (255, 0, 0), 8)
        #
        # dst = cv.line(dst, (700, 1300), (1150, 1100), (255, 0, 0), 8)
        # dst = cv.line(dst, (600, 1300), (650, 1100), (255, 0, 0), 8)

        dst[50:550, 730:1230] = mask
        cv.rectangle(dst, (700, 1400), (600, 1300), (255, 255, 0), 12)
        cv.rectangle(dst, (730, 50), (1230, 550), (255, 255, 0), 12)

        dst = cv.line(dst, (700, 1300), (1230, 550), (255, 255, 0), 12)
        dst = cv.line(dst, (600, 1300), (730, 550), (255, 255, 0), 12)



    # if saved == True:
        # Image.fromarray(dst.astype("uint8")).convert("RGB").save(os.path.join(target, tname))
    io.imshow(dst)
    # plt.xticks([])
    # plt.yticks([])
    # print(dst)
    plt.show()




def version2():
    # the enlarged image on original image.

  #cia
    mode = "cia"

    # img_path = r"D:\codes\dataset\res\resvision\original\cutted\cialandsat338.png"
    # tname = "cia338.png"

    img_path = r"D:\codes\dataset\res\resvision\cia-FSDAF_L2001_338_04dec.png"
    tname = "new-cia-FSDAF_L2001_338_04dec.png"

    img_path = r"D:\codes\dataset\res\resvision\cia-STARFM_L2001_338_04dec.png"
    tname = "new-cia-STARFM_L2001_338_04dec.png"

    img_path = r"D:\codes\dataset\res\resvision\cia-stfnet_L2001_338_04dec.png"
    tname = "new-cia-stfnet_L2001_338_04dec.png"

    img_path = r"D:\codes\dataset\res\resvision\cia-stfmlp_L2001_338_04dec.png"
    tname = "new-cia-stfmlp_L2001_338_04dec.png"


#lgc
    # mode = "lgc"
    #
    # img_path = r"D:\codes\dataset\res\resvision\original\cutted\lgcL20050129_TM.png"
    # tname = r"lgc129-2_TM.png"
    #
    # img_path = r"D:\codes\dataset\res\resvision\lgc-FSDAF_L20050129_TM.tif.png"
    # tname = "new2-lgc-FSDAF_L20050129_TM.tif.png"
    # # #
    # img_path = r"D:\codes\dataset\res\resvision\lgc-STARFM_L20050129_TM.png"
    # tname = "new2-lgc-STARFM_L20050129_TM.png"
    # # # #
    # img_path = r"D:\codes\dataset\res\resvision\lgc-stfnet_L20050129_TM.tif.png"
    # tname = "new2-lgc-stfnet_L20050129_TM.tif.png"
    # # #
    # img_path = r"D:\codes\dataset\res\resvision\lgc-stfmlp_L20050129_TM.tif.png"
    # tname = "new2-lgc-stfmlp_L20050129_TM.tif.png"




    target = r"D:\codes\dataset\res\resvision"


    v2processimage(img_path, target, tname, mode = mode, saved=True)


if __name__ == "__main__":
    # 第一步
    # saveaspng() #选择三个波段存储为PNG

                                        # enlarge() #不用这个

    # 第二步
    version2() #这个是在原图上显示放大的区域