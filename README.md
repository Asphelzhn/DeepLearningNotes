# tensorflow-with-deep-learning-
This repository is for code exercise with "tensorflow with deep learning"

## 4神经网络的优化
1 学习率的设置
学习率既不能过大，也不能过小。为了解决设定学习率的问题， TensorFlow 提供了一种更加灵活的学习率设置方法一一指数衰减法。tf. train.exponential_ decay函数实现了指数衰减学习率。通过这个函数，可以先使用较大的学习率来快速得到一个比较优的解，然后随着迭代的继续逐步减小学习率，使得模型在训练后期更加稳定
''
decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_ steps)
''
