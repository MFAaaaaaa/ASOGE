import pickle


def ReadTxtName(rootdir):
    lines = []
    with open(rootdir, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
    return lines


# my_list = [123, 3.14, "王浩", ["another list"]] #创建一个列表


if __name__ == '__main__':
    resultpath = '../dataset/VISDA-C/visualTess.txt'
    lineslist = {"train_list": []}
    lineslist = ReadTxtName(resultpath)
    # print(type(lineslist), lineslist)
    # with open("visualTes.pkl", 'wb') as fo:  # 将数据写入pkl文件
    #     pickle.dump(lineslist, fo)
    pickle_file = open('Tess.pkl', 'wb')            #创建一个pickle文件，文件后缀名随意,但是打开方式必须是wb（以二进制形式写入）
    pickle.dump(lineslist, pickle_file)                  #将列表倒入文件
    # pickle.close()
