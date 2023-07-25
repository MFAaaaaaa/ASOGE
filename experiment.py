import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import random

# 1.读取数据
path = 'ex2data1.txt'
data = pd.read_csv(path, names=['Exam1', 'Exam2', 'Accepted'])  # 读取数据
data.head()  # 显示数据前五行

# 2.可视化数据集
fig, ax = plt.subplots()  # 此句显示图像
ax.scatter(data[data['Accepted'] == 0]['Exam1'], data[data['Accepted'] == 0]['Exam2'], c='r', marker='x', label='y=0')
ax.scatter(data[data['Accepted'] == 1]['Exam1'], data[data['Accepted'] == 1]['Exam2'], c='b', marker='o', label='y=1')
ax.legend()  # 显示标签
ax.set_xlabel('exam1')  # 设置坐标轴标签
ax.set_ylabel('exam2')

plt.show()


def get_Xy_theta(data):
    data.insert(0, '$x_0$', 1)
    cols = data.shape[1]

    X_ = data.iloc[:, 0:cols - 1]
    X = X_.values

    y_ = data.iloc[:, cols - 1:cols]
    y = y_.values

    return X, y

X, y = get_Xy_theta(data)
theta = np.zeros((X.shape[1], y.shape[1]))

# sigmoid函数
# np.exp(x)表示e的幂运算
def sigmoid(x):
    # 请补充sigmoid激活函数
    y =1/(1+np.exp(-x))
    return y


# 初始化参数
alpha = 0.0001
# 设置不同的迭代次数，观察训练代价的下降
epoch = 2000

# 梯度下降大循环
costs = []
for i in range(epoch):

    # X为训练样本 y为样本标签
    sample_num = X.shape[0]  # 样本数量
    index = random.sample(range(0, sample_num), sample_num)  # 随机序列
    # 随机遍历所有训练样本
    cost_epoch = 0
    for j in range(sample_num):
        temp = index[j]
        x_select = X[temp, :] # 增广矩阵
        x_select = np.expand_dims(x_select, 1)  # 单个训练样本时扩展维度，做矩阵乘法用
        x_select = x_select.T
        y_select = y[temp, :]
        # 逻辑回归输出(提示： 1.线性映射 2.sigmoid函数激活)
        # 1.请在下面补充线性映射 x_select与theta的运算
        f_x_theta =x_select*theta

        # 2.请在sigmoid函数定义部分补充sigmoid
        y_output = sigmoid(f_x_theta)

        # 计算映射函数输出y_output与标签之间的交叉熵代价函数
        # cost = -（y'log(y) + (1-y')log(1-y)）
        # np.log(z)函数
        # 3.请在下面补充交叉熵代价函数
        cost =-(y_select*np.log(y_output)+((1-y_select)*np.log(1-y_output)))
        print(cost)

        # 计算交叉熵代价函数关于参数theta的偏导数
        # 4.请在下面补充偏导数计算
        d_theta=x_select.T*(y_output-y_select)

        # 梯度下降算法更新参数theta
        theta = theta - alpha * d_theta

        cost_epoch = cost_epoch + cost[0]

    costs.append(cost_epoch / sample_num)

#可视化代价函数图像
fig,ax = plt.subplots()
ax.plot(np.arange(epoch), costs) #画直线图
ax.set(xlabel = 'epochs',ylabel='cost',title = 'cost vs epochs')  #设置横纵轴意义
plt.show()

theta_final = theta
print("the final theat:"+"\n"+(str)(theta_final))


# 预测
def predict(X, theta):
    prob = sigmoid(X @ theta)  # 逻辑回归的假设函数
# 5.补充输出判断，若prob大于等于0.5，则返回1；反之，则返回0
#     print(type(prob))
#     print(prob)
    if(prob.any() >= 0.5):return 1
    else:return 0


y_ = np.array(predict(X, theta_final))  # 将预测结果转换为数组
print("the predict:")
# print(y_,type(y_))
y_pre = y_.reshape(y_, 1)

# 求取均值
acc = np.mean(y_pre == y)
print()
print("the acc:"+str(acc))

# 决策边界
# 决策边界就是Xθ=0的时候
coef1 = - theta_final[0, 0] / theta_final[2, 0]
coef2 = - theta_final[1, 0] / theta_final[2, 0]
x = np.linspace(20, 100, 2)
f = coef1 + coef2 * x
fig, ax = plt.subplots()
ax.scatter(data[data['Accepted'] == 0]['Exam1'], data[data['Accepted'] == 0]['Exam2'], c='r', marker='x', label='y=0')
ax.scatter(data[data['Accepted'] == 1]['Exam1'], data[data['Accepted'] == 1]['Exam2'], c='b', marker='o', label='y=1')
ax.legend()
ax.set_xlabel('exam1')
ax.set_ylabel('exam2')
ax.plot(x, f, c='g')
plt.show()
