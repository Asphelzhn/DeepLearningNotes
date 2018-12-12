# tensorflow-with-deep-learning笔记
This repository is for code exercise with "tensorflow with deep learning"

## 4神经网络的优化
1. 学习率的设置

学习率既不能过大，也不能过小。为了解决设定学习率的问题， TensorFlow 提供了一种更加灵活的学习率设置方法一一指数衰减法。tf. train.exponential_ decay函数实现了指数衰减学习率。通过这个函数，可以先使用较大的学习率来快速得到一个比较优的解，然后随着迭代的继续逐步减小学习率，使得模型在训练后期更加稳定。
~~~
decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_ steps)
~~~
其中decayed_learning_ rate 为每一轮优化时使用的学习率， learning_rate 为事先设定的初始学习率， decay_rate 为衰减系数， decay_steps 为衰减速度。。当staircase 的值被设置为True 时， global_step I decay_ steps 会被转化成整数。

2. 过拟合问题

为了避免过拟合问题， 一个非常常用的方法是正则化。正则化的思想就是在损失函数中加入刻画模型复杂程度的指标。假设用于刻画模型在训数据上表现的损失函数为冽的，那么在优化时不是直接优化冽的，而是优化J（θ）＋ λR(w）。其中R(w） 刻画的是模型的复杂程度，而λ 表示模型复杂损失在总损失中的比例。注意这里θ 表示的是一
个神经网络中所有的参数，它包括边上的权重w 和偏置项b 。一般来说模型复杂度只由权重 w 决定。常用的刻画模型复杂度的函数R (w） 有两种， 
一种是LI 正则化，计算公式是：<a href="https://www.codecogs.com/eqnedit.php?latex=$R(w)=\left\|w\right\|_1=\sum_i\left|w_i\right|$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$R(w)=\left\|w\right\|_1=\sum_i\left|w_i\right|$" title="$R(w)=\left\|w\right\|_1=\sum_i\left|w_i\right|$" /></a>

另一种是L2 正则化，计算公式是：<a href="https://www.codecogs.com/eqnedit.php?latex=$R(w)=\left\|w\right\|_2^2=\sum_i\left|w\right|_^2$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$R(w)=\left\|w\right\|_2^2=\sum_i\left|w\right|_^2$" title="$R(w)=\left\|w\right\|_2^2=\sum_i\left|w\right|_^2$" /></a>

无论是哪一种正则化方式，基本的思想都是希望通过限制权重的大小，使得模型不能
任意拟合训练数据中的随机噪音。但这两种正则化的方法也有很大的区别。首先， Ll 正则
化会让参数变得更稀疏，而L2 正则化不会。所谓参数变得更稀疏是指会有更多的参数变为
0 ，这样可以达到类似特征选取的功能。之所以L2 正则化不会让参数变得稀疏的原因是当
参数很小时，比如0.001 ，这个参数的平方基本上就可以忽略了，于是模型不会进一步将这
个参数调整为0 。其次， L I 正则化的计算公式不可导，而L2 正则化公式可导。因为在优
化时需要计算损失函数的偏导数，所以对含有L2 正则化损失函数的优化要更加简洁。优化
带L I 正则化的损失函数要更加复杂，而且优化方法也有很多种。在实践中，也可以将LI
正则化和L2 正则化同时使用：<a href="https://www.codecogs.com/eqnedit.php?latex=$R(w)=\sum_ia\left|w\right|_i&plus;(1-a)w_i^2$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$R(w)=\sum_ia\left|w\right|_i&plus;(1-a)w_i^2$" title="$R(w)=\sum_ia\left|w\right|_i+(1-a)w_i^2$" /></a>

以下代码给出了一个简单的带L2 正则化的损失函数定义：
~~~
w= tf . Variable(tf . random normal([2 , 1) , stddev=l , seed=l))
y = tf.matmul(x , w)
loss= tf.reduce mean(tf.square(y - y)) +
tf.contrib.layers.12_regularizer (lambda) (w)
~~~

第一个部分是均方误差损失函数，它刻画了模型在训练数据上的表现。第二个部分就是正则化，
它防止模型过度模拟训练、数据中的随机噪音。lambda 参数表示了正则化项的权重，也就是
公式J （θ）＋ λ R (w） 中的λ 。w 为需要计算正则化损失的参数。TensorFlow 提供了
tf.contrib. layers.12_regularizer 函数，它可以返回一个函数，这个函数可以计算一个给定参数的L2 正则化项的值。类似的， tf.contrib.layers .l1_regularizer 可以计算LI正则化项的值。
