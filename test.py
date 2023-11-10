import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

x=Variable(torch.linspace(0,100,steps=100).type(torch.FloatTensor))
rand=Variable(torch.rand(100))*50 #加上雜訊
y=x+rand
x_train=x[: -10]
x_test=x[-10 :]
y_train=y[: -10]
y_test=y[-10 :]


#先定義兩個自動微分變數a、b(都是隨機設定值的)
a=Variable(torch.rand(1), requires_grad=True)
b=Variable(torch.rand(1), requires_grad=True)
#設定學習率
learning_rate=0.0001
#完成對a、b的反覆運算
for i in range(1000):
    predictions=a.expand_as(x_train) * x_train + b.expand_as(x_train) #計算在目前a、b條件下的模型預測值
    #a、b的維度都是1，因此需要使用expand_as提升尺寸使與x_train一致
    loss=torch.mean((predictions - y_train)**2) #透過與標籤資料y比較，計算誤差
    print('loss:', loss)
    loss.backward() #對損失函數進行梯度反轉
    a.data.add_(-learning_rate * a.grad.data) #利用上一步計算中獲得a、b的梯度資訊更新原有a、b的data數值
    #.add_()的作用是將原有的a.data數值更新為a.data加上()裡的數值
    b.data.add_(-learning_rate * b.grad.data)
    a.grad.data.zero_() #清空梯度值
    b.grad.data.zero_()
x_data=x_train.data.numpy()
plt.figure(figsize=(10,7)) #定義繪圖視窗
xplot,=plt.plot(x_data, y_train.data.numpy(), 'o') #繪製x,y散點圖
yplot,=plt.plot(x_data, a.data.numpy()*x_data + b.data.numpy()) #繪製擬合直線圖

plt.xlabel('x')
plt.ylabel('y')
str1=str(a.data.numpy()[0]) + 'x +' + str(b.data.numpy()[0]) #將擬合直線參數a、b顯示出來
plt.legend([xplot, yplot], ['Data', str1])
plt.show()