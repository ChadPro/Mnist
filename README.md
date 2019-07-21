# Mnist
### 数据来源
mnist的数据来源于接口**tf.keras.datasets.mnist**, 其中包含60000条train数据， 和10000条test数据。

### 训练
运行  
```
python train_mnist.py
```
另外代码有两种， 一种是卷积， 一种是全连接， 可以自己选择。

### 测试
先运行， 获取一些测试样例， 保存在Demos文件夹下   
```
python get_demos.py
```
运行  
```
python detect_mnist.py
```
可以对应着Demos文件夹下的图片，查看识别结果是否正确
