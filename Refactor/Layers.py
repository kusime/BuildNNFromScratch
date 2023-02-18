import numpy as np

from NormalFunc import cross_entropy_error, softmax

class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self,x):
        # 记录小于零的 tag 为 True
        self.mask = (x <= 0)
        # 复制
        out = x.copy()
        out[self.mask] = 0 # 把小于 0 的转化为0
        return out

    def backward(self,dout):
        dout[self.mask] = 0 
        # 把所有标记的小于0 的点转化为0
        dx = dout
        return dx
    

class Affine:
    """
        仿射变化层
    """
    def __init__(self,W,b):
        # 初始化仿射层
        self.W = W
        self.b = b

        self.dW = None
        self.dB = None
    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W) + self.b
        return  out
    def backward(self,dout):
        # dout 就是 损失函数微分 / 输出 Y的微分
        dX = np.dot(dout,self.W.T) # 传递给后一个层的数据
        self.dW = np.dot(self.x.T,dout) # 本层记录的dW
        self.db = np.sum(dout,axis=0) # 本层记录的 db
        return dX 


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None # one-hot vector

    def forward(self,y,t):
        # 记录标签
        self.t = t
        # print(x[0])
        self.y = softmax(y) # 通过 softmax 后正规化的值 >= 0
        # print("Layer SoftmaxWithLoss (after softmax)",self.y[0])
        # print("Sum",np.sum(self.y[0]))
        # 拿预测和 正确的标签计算交叉熵
        self.loss = cross_entropy_error(self.y,self.t)
        
        return self.loss 
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t)  / batch_size
        return dx