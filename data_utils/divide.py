##深度学习过程中，需要制作训练集和验证集、测试集。

import os, random, shutil


def moveFile(fileDir, tarDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    # rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    rate = 0.03
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    for name in sample:
        shutil.copy(fileDir + name, tarDir + name)
    return

    # if(filenumber * rate < 1):
    #     picknumber = 1
    #     sample = random.sample(pathDir,picknumber)
    #     print(sample)
    #     for name in sample:
    #         shutil.move(fileDir + name, tarDir + name)
    #     return
    # else:
    #     picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    #     sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    #     print(sample)
    #     for name in sample:
    #         shutil.move(fileDir + name, tarDir + name)
    #     return


if __name__ == '__main__':
    dirname = "../dataset/VISDA-C/validation/"
    # fileDir = "../dataset/test/hfu/"  # 源图片文件夹路径
    # tarDir = '../dataset/test_tar/'  # 移动到新的文件夹路径
    for maindir, subdir, file_name_list in os.walk(dirname):
        if (len(subdir) != 0):
            for i in range(len(subdir)):
                pathsub = "../dataset/VISDA-C/target/" + subdir[i]
                os.mkdir(pathsub)
                filrDir = dirname + "/" + subdir[i] + "/"
                tarDir = pathsub + "/"

                # 将图片移回原文件夹
                # tarDir = dirname + "/" + subdir[i] + "/"
                # filrDir = pathsub + "/"

                moveFile(filrDir, tarDir)
