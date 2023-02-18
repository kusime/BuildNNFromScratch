import numpy as np
from collections import OrderedDict

from Layers import Affine, Relu, SoftmaxWithLoss
from NormalFunc import softmax

# np.random.seed(10)
class TwoLayerNetwork:
    """
        input_size: 神经网络的输入
        hidden_size: 神经网络隐藏层
        output_size: 神经网络输出层
    """
    def __init__(self,input_size,hidden_size,output_size,weight_initial_std=1):

        self.params = {}
        # 正规化初始参数
        self.params["W1"] = weight_initial_std * np.random.randn(input_size,hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_initial_std * np.random.randn(hidden_size,output_size)
        self.params["b2"] = np.zeros(output_size)
        
        # generate Layer
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"]) # 仿射 1 层
        self.layers["Relu1"] = Relu() #仿射变化之后的激活层
        self.layers["Affine2"] = Affine(self.params["W2"],self.params["b2"]) # 仿射 2 层
        self.lastLayer = SoftmaxWithLoss() # 输出层

    def predict(self,x):
        """
            正向传播,从输入矩阵到 仿射 2 层输出
        """
        for name,layer in self.layers.items():
            # print("Layer name",name,x.shape)
            x = layer.forward(x)
            # print("Layer name",name,x.shape)
            # input()
        return x
    

    def accuracy(self,input_arrays,reals):
        y = self.predict(input_arrays) # 预测
        y = np.argmax(y,axis=1) # 
        reals = np.argmax(reals,axis=1) # ==
        accuracy = np.sum(y == reals) / float(input_arrays.shape[0])
        return accuracy
    
    def loss(self,x,t):
        """
            计算损失, x 为输入矩阵
            t 为监督数据
        """
        y = self.predict(x) #(N,10)
        # N 个 Affine2 层的输出维度为 10 的分数,没有正规化的

        # Affine2 ---> Softmax with loss 
        # print("Layer SoftmaxWithLoss",y[0])
        softOut = self.lastLayer.forward(y,t)
        # --
        
        return softOut
    

    def gradient(self,x,t):
        # forward
        self.loss(x,t)
        # 反向传播
        dout = 1
        dout = self.lastLayer.backward(dout) # 从SoftmaxWithLoss层开始反向传播
        # print("dout",dout.shape,dout[0])
        # input()

        layers = list(self.layers.values())
        layers.reverse() #反向层
        for layer in layers:
            dout = layer.backward(dout) #开始反向传播并且让层记录对应的微分

        grads = {}
        grads["W1"] = self.layers['Affine1'].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers['Affine2'].dW
        grads["b2"] = self.layers["Affine2"].db

        return grads
    

from  mnist import load_mnist
(x_train, t_train) ,(x_test, t_test) = load_mnist(normalize=True,one_hot_label=True)

iter_num = 10000
tran_size = x_train.shape[0]
batch_size = 100
lr = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(tran_size/batch_size,1)


net = TwoLayerNetwork(input_size=784,hidden_size=200 ,output_size=10)


for i in range(iter_num):
    # 抽取批次
    batch = np.random.choice(tran_size,batch_size)
    x_batch = x_train[batch]
    t_batch = t_train[batch]

    # 使用反向传播运算梯度
    grad = net.gradient(x_batch,t_batch)

    # 更新参数
    for key in ('W1','b1','W2','b2'):
        # grad 是各个参数矩阵的梯度值
        net.params[key] -= lr * grad[key]#梯度下降


    if i % iter_per_epoch == 0:
        train_acc = net.accuracy(x_train,t_train)
        test_acc = net.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    
    loss = net.loss(x_batch,t_batch)
    train_loss_list.append(loss)

# print(np.argmax(softmax(net.predict(np.array([x_test[0]])))),np.array([t_test[0]]))

import matplotlib.pylab as plt
# draw the step activation function
# 损失函数
# x = np.arange(0,len(train_loss_list))
# plt.plot(x,train_loss_list)
x = np.arange(0,len(train_acc_list))
x1 = np.arange(0,len(test_acc_list))
plt.plot(x,train_acc_list,label="train_acc_list")
plt.plot(x1,test_acc_list,label="test_acc_list")

plt.legend()
plt.show()
print(max(train_acc_list))
print(max(test_acc_list))