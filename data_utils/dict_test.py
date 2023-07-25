import pickle

dict_data = {"name": [["张三", 1],["李四", 2]]}

with open("dict_data.pkl", 'wb') as fo:  # 将数据写入pkl文件
    pickle.dump(dict_data, fo)

with open("dict_data.pkl", 'rb') as fo:  # 读取pkl文件数据
    dict_data = pickle.load(fo, encoding='bytes')

# print(dict_data.keys())  # 测试我们读取的文件
print(dict_data)
# print(dict_data["name"])
# == == == == == == == == == == == == == == ==
# 结果如下：
# dict_keys(['name'])
# {'name': ['张三', '李四']}
# ['张三', '李四']

