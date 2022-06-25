# 西安电子科技大学2022届物理与光电工程学院毕业设计
## 2022.4

本代码是本人毕业设计
毕设题目：基于卷积神经网络的红外图像非均匀性校正的研究。


创新点：

·采用一种合并式特征提取单元进行特征提取，从而避免网络层数过多造成的网络退化造成硬件资源浪费

·采取线性噪声模型进行红外图像非均匀性噪声模拟，并且结合残差学习的方式进行级联表示整个网络结构


合并式特征提取单元：

![image](https://user-images.githubusercontent.com/53635655/175773342-c4ac0e84-fd6b-44e2-83b9-69a1f68fdbbc.png)


网络结构:

![image](https://user-images.githubusercontent.com/53635655/175773392-0a5f4922-4cd3-4f01-85d4-fc8157f55d1d.png)


去噪结果：

![image](https://user-images.githubusercontent.com/53635655/175773505-bcc5136c-32b5-4958-9834-1206a22ec5b7.png)
