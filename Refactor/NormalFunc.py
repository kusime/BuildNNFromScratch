import numpy as np


def softmax(x):
    if x.ndim == 2:
        x = x.T # 转置
        x = x - np.max(x,axis=0) # [raw1-max1,raw2-max2] ,对应 | ,第一个维度
        # 这里对应 x = x - np.max(x) # 溢出对策
        y = np.exp(x) / np.sum(np.exp(x),axis=0)  # ,对应 | ,第一个维度

        # 还原维度
        return y.T


    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

# a = np.array([[1010,1000,990],
#               [113,455,788]])
# y=softmax(a)
# print(y)
# print(np.sum(y))
# input()
def cross_entropy_error(pred,real):
    # 计算交叉误差熵
    delta = 1e-7 # avoid overflow
    return -np.sum(real * np.log(pred+delta))