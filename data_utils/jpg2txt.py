import os
import pickle

import pandas as pd


def all_path(dirname1):
    filelistlog = dirname1 + ".txt"  # 保存文件路径
    # filelistlog2 = dirname2 + ".txt"
    # filelistlog = dirname + "\\filelistlog.txt"  # 保存文件路径
    postfix = set(['pdf', 'doc', 'docx', 'epub', 'xlsx', 'djvu', 'chm', 'ppt', 'pptx', 'jpg'])  # 设置要保存的文件格式
    i = -2
    # n = -2
    dictimg = {}
    # dictimg2 = {}
    dictimg['train_list'] = []
    # dictimg2['test_list'] = []

    for maindir, subdir, file_name_list in os.walk(dirname1):
        i += 1
        for filename in file_name_list:
            # apath = str(p[n]) + "/" + str(filename)
            # print(type(os.path.join(maindir, filename)),type(dirname))
            apath = (os.path.join(maindir, filename)).replace(dirname1 + '/', '', 1) # 1是replace的参数，代表只替换1次

            if True:  # 保存全部文件名。若要保留指定文件格式的文件名则注释该句
                # if apath.split('.')[-1] in postfix:   # 匹配后缀，只保存所选的文件格式。若要保存全部文件，则注释该句
                try:
                    with open(filelistlog, 'a+') as fo:

                        fo.writelines(apath + "," + int(i))
                        fo.write('\n')

                except:
                    pass  # 所有异常全部忽略即可
    # for maindir2, subdir2, file_name_list2 in os.walk(dirname2):
    #     n += 1
    #     for filename2 in file_name_list2:
    #         # apath = str(p[n]) + "/" + str(filename)
    #         # print(type(os.path.join(maindir, filename)),type(dirname))
    #         apath2 = (os.path.join(maindir2, filename2)).replace(dirname2 + '/', '', 1) # 1是replace的参数，代表只替换1次
    #
    #         if True:  # 保存全部文件名。若要保留指定文件格式的文件名则注释该句
    #             # if apath.split('.')[-1] in postfix:   # 匹配后缀，只保存所选的文件格式。若要保存全部文件，则注释该句
    #             try:
    #                 with open(filelistlog2, 'a+') as fo:
    #
    #                     fo.writelines(apath2 + "," + str(n))
    #                     fo.write('\n')
    #
    #             except:
    #                 pass  # 所有异常全部忽略即可


    with open(filelistlog, 'r') as f:
        for line in f:
            dictimg['train_list'].append(list(line.strip('\n').split(',')))
    # with open(filelistlog2, 'r') as f2:
    #     for line in f2:
    #         dictimg2['test_list'].append(list(line.strip('\n').split(',')))

    dictimg_all = {}
    dictimg_all.update(dictimg)
    # dictimg_all.update(dictimg2)
    # df = pd.DataFrame([dictimg])
    # df2 = pd.DataFrame([dictimg2])
    # d_all = pd.DataFrame()
    # d_all = d_all.append(df,ignore_index=True)
    # d_all = d_all.append(df2,ignore_index=True)

    with open("strain.pkl", 'wb') as fo:  # 将数据写入pkl文件
        pickle.dump(dictimg_all, fo)




if __name__ == '__main__':
    dirpath1 = "../dataset/VISDA-C/source"  # 指定根目录
    # dirpath2 = "../dataset/Office-31-test/webcam/images"
    all_path(dirpath1)
