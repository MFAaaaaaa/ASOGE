import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



x=[0,1,2,3,4,5,6,7]
y1=[4, 6, 7, 5, 5.5, 6, 9, 7.3]
y2=[2, 2.5, 3.9, 4, 3, 2.4, 8, 6.9]
y3=[3, 4, 6, 4.5, 4, 5, 8.5, 7]
plt.plot(x, y3)
plt.fill_between(x, y1, y2, #上限，下限
        facecolor='blue', #填充颜色
        edgecolor='blue', #边界颜色
        alpha=0.3) #透明度
plt.show()
#
# # 构造一个含有噪声的正弦波
# # time = np.arange(0, 2 * np.pi, 0.1)
# # sin_waves = np.sin(time)
# # print(sin_waves.shape, type(sin_waves))
# # sin_waves = np.expand_dims(sin_waves, axis=-1)
# # print(sin_waves.shape, type(sin_waves))
# #
# # noise = np.random.random((time.size, 10)) - 0.5
# # print('noise shape: ', noise.shape)  # (63, 10)
# # data = sin_waves + noise
# # print(data.shape, type(data))
# # data_mean = np.mean(data, axis=1)
# # data_std = np.std(data, axis=1)
# # data_var = np.var(data, axis=1)
# # data_max = np.max(data, axis=1)
# # data_min = np.min(data, axis=1)
# # plt.figure()
# # plt.plot(data_mean)
# # plt.show()
# # plt.figure()
# # plt.plot(data_std)
# # plt.show()
#
# # 将 time 扩展为 10 列一样的数据
# # time_array = time
# # for i in range(noise.shape[1] - 1):
# #     time_array = np.column_stack((time_array, time))
# #
# # # 将 time 和 signal 平铺为两列数据，且一一对应
# # time_array = time_array.flatten()  # (630,)
# # data = data.flatten()  # (630,)
# # data = np.column_stack((time_array, data))  # (630,2)
# data = np.random.random([2,5])
# print(type(data), data.shape)
# df = pd.DataFrame(data)
# print(type(df), df.shape)
#
# # 绘图
# # sns.relplot(x='time', y='signal', data=df)
# # plt.show()
# sns.relplot(x='time', y='signal', data=df, kind='line')
# plt.show()
