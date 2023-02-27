# 一起学深度学习

# Tutorial
## 一起学《PyTorch深度学习实践》课程
[video](https://www.bilibili.com/video/BV1Y7411d7Ys/?p=1&vd_source=e472d54fbaf4a2a11e9526662ac3a29b)

[homework](https://blog.csdn.net/bit452/category_10569531.html)

代码实现在pytorch-deeplearning-tutorial包下

## 一起学 AI learning
[original github repository](https://github.com/apachecn/ailearning)

代码实现在ai-learning包下

1. 跟着AI learning学习教程进行了代码实现。
2. 使用版本较新的API 
3. 添加了练习使用到的的数据集
4. 使用jupyter-notebook 方便读者学习

## 一起学《动手学深度学习v2》

code in d2l-pytorch-slides package

[video](https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497)

[book](https://zh-v2.d2l.ai/index.html#)

[course](https://course.d2l.ai/zh-v2/)

[d2l-zh-pytorch-slides-github](https://github.com/d2l-ai/d2l-zh-pytorch-slides)

## 一起学deep_thoughts的tutorial

code in deep-thoughts-tutorial package

[video](https://space.bilibili.com/373596439/channel/collectiondetail?sid=57707&ctype=0)

# read and recur papers  
[paper-roadmap-github](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap)

[mli-paper-reading](https://github.com/mli/paper-reading)

[video](https://space.bilibili.com/1567748478/channel/collectiondetail?sid=32744)

tips:
建议使用python-3.8 

在进行论文复现时有用到李沐老师的d2l包

选择合适的版本可以节约大家的时间~


|  paper  |  recurrence  |
|  ----  |  :----:  |
|  [Reducing the dimensionality of data with neural networks.](http://www.cs.toronto.edu/~hinton/absps/science_som.pdf)  |  ⬜  |
|  [A fast learning algorithm for deep belief nets](http://www.cs.toronto.edu/~hinton/absps/ncfast.pdf)  |  ⬜  |
|  [Deep learning-Three Giants' Survey](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)  |  ❌  | 

## 自然语言处理 - Transformer
|  年份 | 名字                                                         | 简介                 |  recurrence  |
| ---- | ------------------------------------------------------------ | -------------------- |  :----:  |
| 2017 | [Transformer](https://arxiv.org/pdf/1706.03762.pdf)  | 	继MLP、CNN、RNN后的第四大类架构 |  ⬜  |
| 2018 | [BERT](https://arxiv.org/pdf/1810.04805.pdf)  | Transformer一统NLP的开始 |  ⬜  |

## 计算机视觉 - CNN
|  年份 | 名字                                                         | 简介                 |  recurrence  |
| ---- | ------------------------------------------------------------ | -------------------- |  :----:  |
| 2012 | [AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)  | 深度学习热潮的奠基作 |  [✅](https://github.com/xiamingtx/DeepLearningTutorial/blob/main/papers-recurrence/AlexNet.ipynb)  |
| 2015 | [ResNet](https://arxiv.org/pdf/1512.03385.pdf)  | 	构建深层网络都要有的残差连接。 |  [✅](https://github.com/xiamingtx/DeepLearningTutorial/blob/main/papers-recurrence/ResNet.ipynb)  |

## 计算机视觉 - Transformer
|  年份 | 名字                                                         | 简介                 |  recurrence  |
| ---- | ------------------------------------------------------------ | -------------------- |  :----:  |
| 2020 | [ViT](https://arxiv.org/pdf/2010.11929.pdf)  | Transformer杀入CV界 |  ⬜  |
| 2021 | [MAE](https://arxiv.org/pdf/2111.06377.pdf)  | 	BERT的CV版 |  ⬜  |

## 生成模型
|  年份 | 名字                                                         | 简介                 |  recurrence  |
| ---- | ------------------------------------------------------------ | -------------------- |  :----:  |
| 2014 | [GAN](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)  | 	生成模型的开创工作 |  [✅](https://github.com/xiamingtx/DeepLearningTutorial/blob/main/papers-recurrence/GAN.ipynb)  |

## 对比学习
|  年份 | 名字                                                         | 简介                 |  recurrence  |
| ---- | ------------------------------------------------------------ | -------------------- |  :----:  |
| 2018 | [InstDisc](https://arxiv.org/pdf/1805.01978.pdf) | 提出实例判别和memory bank做对比学习  |  ⬜  |            
| 2018 | [CPC](https://arxiv.org/pdf/1807.03748.pdf) | 对比预测编码，图像语音文本强化学习全都能做   |  ⬜  |              
| 2019 | [InvaSpread](https://arxiv.org/pdf/1904.03436.pdf) | 一个编码器的端到端对比学习  |  ⬜  |           
| 2019 |  [CMC](https://arxiv.org/pdf/1906.05849.pdf) | 多视角下的对比学习  |  ⬜  |              
| 2019 | [MoCov1](https://arxiv.org/pdf/1911.05722.pdf) | 无监督训练效果也很好  |  ⬜  |                  
| 2020 |  [SimCLRv1](https://arxiv.org/pdf/2002.05709.pdf) |  简单的对比学习 (数据增强 + MLP head + 大batch训练久)  |  ⬜  |          
| 2020 | [MoCov2](https://arxiv.org/pdf/2003.04297.pdf) | MoCov1 + improvements from SimCLRv1  |  ⬜  |              
| 2020 |  [SimCLRv2](https://arxiv.org/pdf/2006.10029.pdf) | 大的自监督预训练模型很适合做半监督学习  |  ⬜  |        
| 2020 |  [BYOL](https://arxiv.org/pdf/2006.07733.pdf) | 不需要负样本的对比学习  |  ⬜  |                   
| 2020 |  [SWaV](https://arxiv.org/pdf/2006.09882.pdf) | 聚类对比学习  |  ⬜  |                   
| 2020 |  [SimSiam](https://arxiv.org/pdf/2011.10566.pdf) | 化繁为简的孪生表征学习  |  ⬜  |              
| 2021 | [MoCov3](https://arxiv.org/pdf/2104.02057.pdf) | 如何更稳定的自监督训练ViT  |  ⬜  |           
| 2021 |  [DINO](https://arxiv.org/pdf/2104.14294.pdf) | transformer加自监督在视觉也很香  |  ⬜  |             

## 图神经网络
|  年份 | 名字                                                         | 简介                 |  recurrence  |
| ---- | ------------------------------------------------------------ | -------------------- |  :----:  |
| 2021 | [图神经网络介绍](https://distill.pub/2021/gnn-intro/)  | 	GNN的可视化介绍 |  ⬜  |

# 一起学习~ 有问题欢迎指出