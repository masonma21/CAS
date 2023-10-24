## Week 1：深度学习与Paddle入门

### 入门深度学习、环境搭建

机器学习算法理论在上个世纪90年代发展成熟，在许多领域都取得了成功。但平静的日子只延续到2010年左右，随着大数据的涌现和计算机算力提升，深度学习模型异军突起，极大改变了机器学习的应用格局。今天，多数机器学习任务都可以使用深度学习模型解决，尤其在语音、计算机视觉和自然语言处理等领域，深度学习模型的效果比传统机器学习算法有显著提升。

那么相比传统的机器学习算法，深度学习做出了哪些改进呢？其实**两者在理论结构上是一致的，即：模型假设、评价函数和优化算法，其根本差别在于假设的复杂度**，如 **图6** 所示。

![img](https://ai-studio-static-online.cdn.bcebos.com/f7571e63a908401385013c15ecd15e92eff94d2d1e624805a1815b0385faa55b)


图7：深度学习的模型复杂度难以想象





不是所有的任务都像牛顿第二定律那样简单直观。对于 **图7** 中的美女照片，人脑可以接收到五颜六色的光学信号，能用极快的速度反应出这张图片是一位美女，而且是程序员喜欢的类型。但对计算机而言，只能接收到一个数字矩阵，对于美女这种高级的语义概念，从像素到高级语义概念中间要经历的信息变换的复杂性是难以想象的！这种变换已经无法用数学公式表达，因此研究者们借鉴了人脑神经元的结构，设计出神经网络的模型。



##### 神经网络的基本概念

人工神经网络包括多个神经网络层，如卷积层、全连接层、LSTM等，每一层又包括很多神经元，超过三层的非线性神经网络都可以被称为深度神经网络。通俗的讲，深度学习的模型可以视为是输入到输出的映射函数，如图像到高级语义（美女）的映射，足够深的神经网络理论上可以拟合任何复杂的函数。因此神经网络非常适合学习样本数据的内在规律和表示层次，对文字、图像和语音任务有很好的适用性。因为这几个领域的任务是人工智能的基础模块，所以深度学习被称为实现人工智能的基础也就不足为奇了。

神经网络结构如 **图8** 所示。

![img](https://ai-studio-static-online.cdn.bcebos.com/af79017f3e1143fab258386460c324c4adf7ab0a51364fa98474d04798721752)


图8：神经网络结构示意图





- 神经元：


  神经网络中每个节点称为神经元，由两部分组成：

  - 加权和：将所有输入加权求和。
  - 非线性变换（激活函数）：加权和的结果经过一个非线性函数变换，让神经元计算具备非线性的能力。

- **多层连接：** 大量这样的节点按照不同的层次排布，形成多层的结构连接起来，即称为神经网络。

- **前向计算：** 从输入计算输出的过程，顺序从网络前至后。

- **计算图：** 以图形化的方式展现神经网络的计算逻辑又称为计算图。我们也可以将神经网络的计算图以公式的方式表达，如下：

$$
Y =f_3 ( f_2 ( f_1 ( w_1\cdot x_1+w_2\cdot x_2+w_3\cdot x_3+b ) + … ) … ) … )
$$

由此可见，神经网络并没有那么神秘，它的本质是一个含有很多参数的“大公式”。如果大家感觉这些概念仍过于抽象，理解的不够透彻，先不用着急，后续我们会以实践案例的方式，再次介绍这些概念。



##### 深度学习的发展历程

那么如何设计神经网络呢？下一章会以“房价预测”为例，演示使用Python实现神经网络模型的细节。在此之前，我们先回顾下深度学习的悠久历史。

神经网络思想的提出已经是75年前的事情了，现今的神经网络和深度学习的设计理论是一步步趋于完善的。在这漫长的发展岁月中，一些取得关键突破的闪光时刻，值得我们这些深度学习爱好者们铭记，如 **图9** 所示。

![img](https://ai-studio-static-online.cdn.bcebos.com/8212741e9a70495ea467a2d2a861baff9ecd964aa67447e8a415b162dd5d01a4)


图9：深度学习发展历程





- **1940年代**：首次提出神经元的结构，但权重是不可学的。
- **50-60年代**：提出权重学习理论，神经元结构趋于完善，开启了神经网络的第一个黄金时代。
- **1969年**：提出异或问题（人们惊讶的发现神经网络模型连简单的异或问题也无法解决，对其的期望从云端跌落到谷底），神经网络模型进入了被束之高阁的黑暗时代。
- **1986年**：新提出的多层神经网络解决了异或问题，但随着90年代后理论更完备并且实践效果更好的SVM等机器学习模型的兴起，神经网络并未得到重视。
- **2010年左右**：深度学习进入真正兴起时期。随着神经网络模型改进的技术在语音和计算机视觉任务上大放异彩，也逐渐被证明在更多的任务，如自然语言处理以及海量数据的任务上更加有效。至此，神经网络模型重新焕发生机，并有了一个更加响亮的名字：深度学习。

为何神经网络到2010年后才焕发生机呢？这与深度学习成功所依赖的先决条件：大数据涌现、硬件发展和算法优化有关。

- **大数据是神经网络发展的有效前提**。神经网络和深度学习是非常强大的模型，需要足够量级的训练数据。时至今日，之所以很多传统机器学习算法和人工特征依然是足够有效的方案，原因在于很多场景下没有足够的标记数据来支撑深度学习。深度学习的能力特别像科学家阿基米德的豪言壮语：“给我一根足够长的杠杆，我能撬动地球！”。深度学习也可以发出类似的豪言：“给我足够多的数据，我能够学习任何复杂的关系”。但在现实中，足够长的杠杆与足够多的数据一样，往往只能是一种美好的愿景。直到近些年，各行业IT化程度提高，累积的数据量爆发式地增长，才使得应用深度学习模型成为可能。
- **依靠硬件的发展和算法的优化**。现阶段，依靠更强大的计算机、GPU、autoencoder预训练和并行计算等技术，深度学习在模型训练上的困难已经被逐渐克服。其中，数据量和硬件是更主要的原因。没有前两者，科学家们想优化算法都无从进行。



##### 深度学习的研究和应用蓬勃发展

早在1998年，一些科学家就已经使用神经网络模型识别手写数字图像了。但深度学习在计算机视觉应用上的兴起，还是在2012年ImageNet比赛上，使用AlexNet做图像分类。如果比较下1998年和2012年的模型，会发现两者在网络结构上非常类似，仅在细节上有所优化。在这十四年间，计算性能的大幅提升和数据量的爆发式增长，促使模型完成了从“简单的数字识别”到“复杂的图像分类”的跨越。

虽然历史悠久，但深度学习在今天依然在蓬勃发展，一方面基础研究快速发展，另一方面工业实践层出不穷。基于深度学习的顶级会议ICLR(International Conference on Learning Representations)统计，深度学习相关的论文数量呈逐年递增的状态，如 **图10** 所示。同时，不仅仅是深度学习会议，与数据和模型技术相关的会议ICML和KDD，专注视觉的CVPR和专注自然语言处理的EMNLP等国际会议的大量论文均涉及着深度学习技术。该领域和相关领域的研究方兴未艾，技术仍在不断创新突破中。

![img](https://ai-studio-static-online.cdn.bcebos.com/217f355d70994f55bf74f590d365ef70989916730e3b4077ad96b555aa148469)


图10：深度学习相关论文数量逐年攀升





另一方面，以深度学习为基础的人工智能技术，在升级改造众多的传统行业领域，存在极其广阔的应用场景。**图11** 选自艾瑞咨询的研究报告，人工智能技术不仅可在众多行业中落地应用（广度），同时，在部分行业（如安防）已经实现了市场化变现和高速增长（深度），为社会贡献了巨大的经济价值。

![img](https://ai-studio-static-online.cdn.bcebos.com/01d970aa13294bbda637027b98c24d0472ab8b7c2a9a4c6bb69610ec76ebdc80)


图11：以深度学习为基础的AI技术在各行业广泛应用



##### 深度学习改变了AI应用的研发模式

###### 实现了端到端的学习

深度学习改变了很多领域算法的实现模式。在深度学习兴起之前，很多领域建模的思路是投入大量精力做特征工程，将专家对某个领域的“人工理解”沉淀成特征表达，然后使用简单模型完成任务（如分类或回归）。而在数据充足的情况下，深度学习模型可以实现端到端的学习，即不需要专门做特征工程，将原始的特征输入模型中，模型可同时完成特征提取和分类任务，如 **图12** 所示。

![img](https://ai-studio-static-online.cdn.bcebos.com/df67d165d79b448e93a097cd390211c564248d76b03745378b465c739914d9f6)


图12：深度学习实现了端到端的学习



以计算机视觉任务为例，特征工程是诸多图像科学家基于人类对视觉理论的理解，设计出来的一系列提取特征的计算步骤，典型如SIFT特征。在2010年之前的计算机视觉领域，人们普遍使用SIFT一类特征+SVM一类的简单浅层模型完成建模任务。

> **说明：**
>
> SIFT特征由David Lowe在1999年提出，在2004年加以完善。SIFT特征是基于物体上的一些局部外观的兴趣点而与影像的大小和旋转无关。对于光线、噪声、微视角改变的容忍度也相当高。基于这些特性，它们是高度显著而且相对容易撷取，在母数庞大的特征数据库中，很容易辨识物体而且鲜有误认。使用SIFT特征描述对于部分物体遮蔽的侦测率也相当高，甚至只需要3个以上的SIFT物体特征就足以计算出位置与方位。在现今的电脑硬件速度下和小型的特征数据库条件下，辨识速度可接近即时运算。SIFT特征的信息量大，适合在海量数据库中快速准确匹配。



###### 实现了深度学习框架标准化

除了应用广泛的特点外，深度学习还推动人工智能进入工业大生产阶段，算法的通用性导致标准化、自动化和模块化的框架产生，如 **图13** 所示。

![img](https://ai-studio-static-online.cdn.bcebos.com/091cd30d8cc04b0d86d5cf21bf9478fa2e0dcb8f4e8245d0812f6d1209d083b9)


图13：深度学习模型具有通用性特点



在此之前，不同流派的机器学习算法理论和实现均不同，导致每个算法均要独立实现，如随机森林和支撑向量机（SVM）。但在深度学习框架下，不同模型的算法结构有较大的通用性，如常用于计算机视觉的卷积神经网络模型（CNN）和常用于自然语言处理的长期短期记忆模型(LSTM)，都可以分为组网模块、梯度下降的优化模块和预测模块等。这使得抽象出统一的框架成为了可能，并大大降低了编写建模代码的成本。一些相对通用的模块，如网络基础算子的实现、各种优化算法等都可以由框架实现。建模者只需要关注数据处理，配置组网的方式，以及用少量代码串起训练和预测的流程即可。

在深度学习框架出现之前，机器学习工程师处于“手工业作坊”生产的时代。为了完成建模，工程师需要储备大量数学知识，并为特征工程工作积累大量行业知识。每个模型是极其个性化的，建模者如同手工业者一样，将自己的积累形成模型的“个性化签名”。而今，“深度学习工程师”进入了工业化大生产时代。只要掌握深度学习必要但少量的理论知识，掌握Python编程，即可在深度学习框架上实现非常有效的模型，甚至与该领域最领先的模型不相上下。建模这个被“老科学家”们长期把持的建模领域面临着颠覆，也是新入行者的机遇。

在深度学习框架出现之前，机器学习工程师处于“孤军作战”的时代。为了完成建模，工程师需要储备大量数学知识，并为特征工程工作积累大量行业知识，才能出色的完成任务。而今，深度学习工程师进入了“兵团作战”的时代，大量复杂的专业和行业知识不再成为必须，只要掌握深度学习必要但少量的理论知识，掌握Python编程，即可在深度学习框架上实现非常有效的模型，甚至与该领域最领先的模型不相上下。建模这个入行很难的领域，开始向AI新入行者敞开怀抱，成为更多AI开发者的新机遇。

![img](https://ai-studio-static-online.cdn.bcebos.com/64b0d7a2875d4309b51128487d6dc03c791eb0c49a654ccda15255bbfc63c7a7)


图14：深度学习框架大大减低了AI建模难度



人生天地之间，若白驹过隙，忽然而已，每个人都希望留下自己的足迹。为何要学习深度学习技术，以及如何通过这本书来学习呢？一方面，深度学习的应用前景广阔，是极好的发展方向和职业选择。

#### 任务

搭建好基于conda+vscode+paddle的运行环境，如果机器性能较差，则可以借助一些云计算资源平台，比如Google colab、百度官方的aistudio完成后续学习。

---



### 使用Python语言和Numpy库来构建神经网络模型

#### 波士顿房价预测任务

上一节我们初步认识了神经网络的基本概念（如神经元、多层连接、前向计算、计算图）和模型结构三要素（模型假设、评价函数和优化算法）。本节将以“波士顿房价预测”任务为例，向读者介绍使用Python语言和Numpy库来构建神经网络模型的思考过程和操作方法。

波士顿房价预测是一个经典的机器学习任务，类似于程序员世界的“Hello World”。和大家对房价的普遍认知相同，波士顿地区的房价受诸多因素影响。该数据集统计了13种可能影响房价的因素和该类型房屋的均价，期望构建一个基于13个因素进行房价预测的模型，如 **图1** 所示。

![img](https://ai-studio-static-online.cdn.bcebos.com/abce0cb2a92f4e679c6855cfa520491597171533a0b0447e8d51d904446e213e)


图1：波士顿房价影响因素示意图

对于预测问题，可以根据预测输出的类型是连续的实数值，还是离散的标签，区分为回归任务和分类任务。因为房价是一个连续值，所以房价预测显然是一个回归任务。下面我们尝试用最简单的线性回归模型解决这个问题，并用神经网络来实现这个模型。



#### 线性回归模型

假设房价和各影响因素之间能够用线性关系来描述：
$$
y = {\sum_{j=1}^Mx_j w_j} + b
$$
模型的求解即是通过数据拟合出每个$w_j$和$b$。其中，$w_j$和$b$分别表示该线性模型的权重和偏置。一维情况下，$w_j$和$b$是直线的斜率和截距。

线性回归模型使用均方误差作为损失函数（Loss），用以衡量预测房价和真实房价的差异，公式如下：
$$
MSE = \frac{1}{n} \sum_{i=1}^n(\hat{Y_i} - {Y_i})^{2}
$$

> **思考：**
>
> 为什么要以均方误差作为损失函数？即将模型在每个训练样本上的预测误差加和，来衡量整体样本的准确性。这是因为损失函数的设计不仅仅要考虑“合理性”，同样需要考虑“易解性”，这个问题在后面的内容中会详细阐述。

------



#### 线性回归模型的神经网络结构

神经网络的标准结构中每个神经元由加权和与非线性变换构成，然后将多个神经元分层的摆放并连接形成神经网络。线性回归模型可以认为是神经网络模型的一种极简特例，是一个只有加权和、没有非线性变换的神经元（无需形成网络），如 **图2** 所示。

![img](https://ai-studio-static-online.cdn.bcebos.com/f9117a5a34d44b1eab85147e62b4e6295e485e48d79d4a03adaa14a447ffd230)


图2：线性回归模型的神经网络结构



#### 构建波士顿房价预测任务的神经网络模型

深度学习不仅实现了模型的端到端学习，还推动了人工智能进入工业大生产阶段，产生了标准化、自动化和模块化的通用框架。不同场景的深度学习模型具备一定的通用性，五个步骤即可完成模型的构建和训练，如 **图3** 所示。

![img](https://ai-studio-static-online.cdn.bcebos.com/12fdca24a3b94166a9e8c815ef4b0e4ddfec541f3a024a4392f1fc17fa186c7b)


图3：构建神经网络/深度学习模型的基本步骤

正是由于深度学习的建模和训练的过程存在通用性，在构建不同的模型时，只有模型三要素不同，其它步骤基本一致，深度学习框架才有用武之地。



#### 数据处理

Numpy介绍：https://www.paddlepaddle.org.cn/tutorials/projectdetail/3145000

数据处理包含五个部分：数据导入、数据形状变换、数据集划分、数据归一化处理和封装`load data`函数。数据预处理后，才能被模型调用。

> **说明：**
>
> - 本教程中的代码都可以在AI Studio上直接运行，Print结果都是基于程序真实运行的结果。
>
> - 由于是真实案例，代码之间存在依赖关系，因此需要读者逐条、全部运行，否则会导致命令执行报错。




##### 读入数据

通过如下代码读入数据，了解下波士顿房价的数据集结构，数据存放在本地目录下housing.data文件中。

```python
# 导入需要用到的package
import numpy as np
import json
# 读入训练数据
datafile = './work/housing.data'
data = np.fromfile(datafile, sep=' ')
data
```

```
array([6.320e-03, 1.800e+01, 2.310e+00, ..., 3.969e+02, 7.880e+00, 1.190e+01])
```



##### 数据形状变换

由于读入的原始数据是1维的，所有数据都连在一起。因此需要我们将数据的形状进行变换，形成一个2维的矩阵，每行为一个数据样本（14个值），每个数据样本包含13个XX*X*（影响房价的特征）和一个YY*Y*（该类型房屋的均价）。

```python
# 读入之后的数据被转化成1维array，其中array的第0-13项是第一条数据，第14-27项是第二条数据，以此类推.... 
# 这里对原始数据做reshape，变成N x 14的形式
feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
feature_num = len(feature_names)
data = data.reshape([data.shape[0] // feature_num, feature_num])
# 查看数据
x = data[0]
print(x.shape)
print(x)
```

```
(14,)
[6.320e-03 1.800e+01 2.310e+00 0.000e+00 5.380e-01 6.575e+00 6.520e+01
 4.090e+00 1.000e+00 2.960e+02 1.530e+01 3.969e+02 4.980e+00 2.400e+01]
```



##### 数据集划分

将数据集划分成训练集和测试集，其中训练集用于确定模型的参数，测试集用于评判模型的效果。为什么要对数据集进行拆分，而不能直接应用于模型训练呢？这与学生时代的授课和考试关系比较类似，如 **图4** 所示。

![img](https://ai-studio-static-online.cdn.bcebos.com/a1c845a50e28474d9aa72028edfea33f1a3deca1d54d40ec94ba366d3a18c408)


图4：训练集和测试集拆分的意义

上学时总有一些自作聪明的同学，平时不认真学习，考试前临阵抱佛脚，将习题死记硬背下来，但是成绩往往并不好。因为学校期望学生掌握的是知识，而不仅仅是习题本身。另出新的考题，才能鼓励学生努力去掌握习题背后的原理。同样我们期望模型学习的是任务的本质规律，而不是训练数据本身，模型训练未使用的数据，才能更真实的评估模型的效果。

在本案例中，我们将80%的数据用作训练集，20%用作测试集，实现代码如下。通过打印训练集的形状，可以发现共有404个样本，每个样本含有13个特征和1个预测值。

```python
ratio = 0.8
offset = int(data.shape[0] * ratio)
training_data = data[:offset]
training_data.shape
```

```
(404, 14)
```



##### 数据归一化处理

对每个特征进行归一化处理，使得每个特征的取值缩放到0~1之间。这样做有两个好处：一是模型训练更高效；二是特征前的权重大小可以代表该变量对预测结果的贡献度（因为每个特征值本身的范围相同）。

```python
# 计算train数据集的最大值，最小值，平均值
maximums, minimums, avgs = \
                     training_data.max(axis=0), \
                     training_data.min(axis=0), \
     training_data.sum(axis=0) / training_data.shape[0]
# 对数据进行归一化处理
for i in range(feature_num):
    #print(maximums[i], minimums[i], avgs[i])
    data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])
```



##### 封装成load data函数

将上述几个数据处理操作封装成`load data`函数，以便下一步模型的调用，实现方法如下。

```python
def load_data():
    # 从文件导入数据
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算训练集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化处理
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data
# 获取数据
training_data, test_data = load_data()
x = training_data[:, :-1]
y = training_data[:, -1:]
# 查看数据
print(x[0])
print(y[0])
```

```
[0.         0.18       0.07344184 0.         0.31481481 0.57750527
 0.64160659 0.26920314 0.         0.22755741 0.28723404 1.
 0.08967991]
[0.42222222]
```



#### 模型设计

模型设计是深度学习模型关键要素之一，也称为网络结构设计，相当于模型的假设空间，即实现模型“前向计算”（从输入到输出）的过程。

如果将输入特征和输出预测值均以向量表示，输入特征$x$有13个分量，$y$有1个分量，那么参数权重的形状（shape）是$13\times1$。假设我们以如下任意数字赋值参数做初始化：
$$
w=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.0]
$$

```python
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.0]
w = np.array(w).reshape([13, 1])
```

取出第1条样本数据，观察样本的特征向量与参数向量相乘的结果。

```python
x1=x[0]
t = np.dot(x1, w)
print(t)
```

```
[0.69474855]
```

完整的线性回归公式，还需要初始化偏移量$b$​，同样随意赋初值-0.2。那么，线性回归模型的完整输出是$z=t+b$​，这个从特征和参数计算输出值的过程称为“前向计算”。

```python
b = -0.2
z = t + b
print(z)
```

```
[0.49474855]
```

将上述计算预测输出的过程以“类和对象”的方式来描述，类成员变量有参数$w$​和$b$​。通过写一个`forward`函数（代表“前向计算”）完成上述从特征和参数到输出预测值的计算过程，代码如下所示。

```python
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
        
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
```

基于Network类的定义，模型的计算过程如下所示。

```python
net = Network(13)
x1 = x[0]
y1 = y[0]
z = net.forward(x1)
print(z)
```

```
[2.39362982]
```



#### 训练配置

模型设计完成后，需要通过训练配置寻找模型的最优值，即通过损失函数来衡量模型的好坏。训练配置也是深度学习模型关键要素之一。

通过模型计算$x_1$表示的影响因素所对应的房价应该是$z$, 但实际数据告诉我们房价是$y$。这时我们需要有某种指标来衡量预测值$z$跟真实值$y$之间的差距。对于回归问题，最常采用的衡量方法是使用均方误差作为评价模型好坏的指标，具体定义如下：
$$
Loss = (y - z)^2
$$
上式中的Loss（简记为: L）通常也被称作损失函数，它是衡量模型好坏的指标。在回归问题中，均方误差是一种比较常见的形式，分类问题中通常会采用交叉熵作为损失函数，在后续的章节中会更详细的介绍。对一个样本计算损失函数值的实现如下：

```python
Loss = (y1 - z)*(y1 - z)
print(Loss)
```

```
[3.88644793]
```

因为计算损失函数时需要把每个样本的损失函数值都考虑到，所以我们需要对单个样本的损失函数进行求和，并除以样本总数$N$​。
$$
Loss= \frac{1}{N}\sum_{i=1}^N{(y_i - z_i)^2}
$$
在Network类下面添加损失函数的计算过程如下：

```python
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
        
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    def loss(self, z, y):
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return cost
```

使用定义的Network类，可以方便的计算预测值和损失函数。需要注意的是，类中的变量$x$, $w$，$b$, $z$, $error$等均是向量。以变量xx*x*为例，共有两个维度，一个代表特征数量（值为13），一个代表样本数量，代码如下所示。

```python
net = Network(13)
# 此处可以一次性计算多个样本的预测值和损失函数
x1 = x[0:3]
y1 = y[0:3]
z = net.forward(x1)
print('predict: ', z)
loss = net.loss(z, y1)
print('loss:', loss)
```

```
predict:  [[2.39362982]
 [2.46752393]
 [2.02483479]]
loss: 3.384496992612791
```



#### 训练过程

上述计算过程描述了如何构建神经网络，通过神经网络完成预测值和损失函数的计算。接下来介绍如何求解参数$w$和$b$的数值，这个过程也称为模型训练过程。训练过程是深度学习模型的关键要素之一，其目标是让定义的损失函数$Loss$尽可能的小，也就是说找到一个参数解$w$和$b$，使得损失函数取得极小值。

我们先做一个小测试：如 **图5** 所示，基于微积分知识，求一条曲线在某个点的斜率等于函数在该点的导数值。那么大家思考下，当处于曲线的极值点时，该点的斜率是多少？

![img](https://ai-studio-static-online.cdn.bcebos.com/94f0437e6a454a0682f3b831c96a62bdaf40898af25145ec9b5b50bc80391f5c)


图5：曲线斜率等于导数值



这个问题并不难回答，处于曲线极值点时的斜率为0，即函数在极值点的导数为0。那么，让损失函数取极小值的$w$和$b$应该是下述方程组的解：
$$
\frac{\partial{L}}{\partial{w}}=0 \\
\frac{\partial{L}}{\partial{b}}=0
$$
将样本数据$(x,y)$带入上面的方程组中即可求解出$w$和$b$的值，但是这种方法只对线性回归这样简单的任务有效。如果模型中含有非线性变换，或者损失函数不是均方差这种简单的形式，则很难通过上式求解。为了解决这个问题，下面我们将引入更加普适的数值求解方法：梯度下降法。



#### 梯度下降法

在现实中存在大量的函数正向求解容易，但反向求解较难，被称为单向函数，这种函数在密码学中有大量的应用。密码锁的特点是可以迅速判断一个密钥是否是正确的(已知x，求y很容易)，但是即使获取到密码锁系统，无法破解出正确的密钥是什么（已知y，求x很难）。

这种情况特别类似于一位想从山峰走到坡谷的盲人，他看不见坡谷在哪（无法逆向求解出Loss导数为0时的参数值），但可以伸脚探索身边的坡度（当前点的导数值，也称为梯度）。那么，求解Loss函数最小值可以这样实现：从当前的参数取值，一步步的按照下坡的方向下降，直到走到最低点。这种方法笔者称它为“盲人下坡法”。哦不，有个更正式的说法“梯度下降法”。

训练的关键是找到一组(w,b)，使得损失函数L取极小值。我们先看一下损失函数L只随两个参数$w_5$、$w_9$变化时的简单情形，启发下寻解的思路。
$$
L=L(w_5, w_9)
$$
这里我们将$w_0, w_1, ..., w_{12}$中除$w_5, w_9$之外的参数和$b$都固定下来，可以用图画出$L(w_5, w_9)$的形式。

```python
net = Network(13)
losses = []
#只画出参数w5和w9在区间[-160, 160]的曲线部分，以及包含损失函数的极值
w5 = np.arange(-160.0, 160.0, 1.0)
w9 = np.arange(-160.0, 160.0, 1.0)
losses = np.zeros([len(w5), len(w9)])

#计算设定区域内每个参数取值所对应的Loss
for i in range(len(w5)):
    for j in range(len(w9)):
        net.w[5] = w5[i]
        net.w[9] = w9[j]
        z = net.forward(x)
        loss = net.loss(z, y)
        losses[i, j] = loss

#使用matplotlib将两个变量和对应的Loss作3D图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)

w5, w9 = np.meshgrid(w5, w9)

ax.plot_surface(w5, w9, losses, rstride=1, cstride=1, cmap='rainbow')
plt.show()
```

对于这种简单情形，我们利用上面的程序，可以在三维空间中画出损失函数随参数变化的曲面图。从图中可以看出有些区域的函数值明显比周围的点小。

需要说明的是：为什么这里我们选择$w_5$和$w_9$来画图？这是因为选择这两个参数的时候，可比较直观的从损失函数的曲面图上发现极值点的存在。其他参数组合，从图形上观测损失函数的极值点不够直观。

观察上述曲线呈现出“圆滑”的坡度，这正是我们选择以均方误差作为损失函数的原因之一。**图6** 呈现了只有一个参数维度时，均方误差和绝对值误差（只将每个样本的误差累加，不做平方处理）的损失函数曲线图。

```
<Figure size 640x480 with 1 Axes>
```

![img](https://ai-studio-static-online.cdn.bcebos.com/99487dca6520441db5073d1c154b5d2fb1174b5cf4d946c29f9d80a209bc2687)


图6：均方误差和绝对值误差损失函数曲线图

由此可见，均方误差表现的“圆滑”的坡度有两个好处：

- 曲线的最低点是可导的。
- 越接近最低点，曲线的坡度逐渐放缓，有助于通过当前的梯度来判断接近最低点的程度（是否逐渐减少步长，以免错过最低点）。

而绝对值误差是不具备这两个特性的，这也是损失函数的设计不仅仅要考虑“合理性”，还要追求“易解性”的原因。

现在我们要找出一组$[w5,w9]$的值，使得损失函数最小，实现梯度下降法的方案如下：

- 步骤1：随机的选一组初始值，例如：$[w5,w9]=[−100.0,−100.0]$
- 步骤2：选取下一个点$[w_5^{'} , w_9^{'}]$，使得$L(w_5^{'} , w_9^{'}) < L(w_5, w_9)$
- 步骤3：重复步骤2，直到损失函数几乎不再下降。

如何选择$[w_5^{'} , w_9^{'}]$是至关重要的，第一要保证$L$是下降的，第二要使得下降的趋势尽可能的快。微积分的基础知识告诉我们，沿着梯度的反方向，是函数值下降最快的方向，如 **图7** 所示。简单理解，函数在某一个点的梯度方向是曲线斜率最大的方向，但梯度方向是向上的，所以下降最快的是梯度的反方向。

![img](https://ai-studio-static-online.cdn.bcebos.com/5f8322f6172542dab0f78684b70efe45d819895332af4cabb7c536217ab0bb26)


图7：梯度下降方向示意图



##### 计算梯度

上面我们讲过了损失函数的计算方法，这里稍微改写，为了使梯度计算更加简洁，引入因子$\frac{1}{2}$，定义损失函数如下：

$$
L= \frac{1}{2N}\sum_{i=1}^N{(y_i - z_i)^2}
$$
其中$z_i$是网络对第$i$个样本的预测值：

$$
z_i = \sum_{j=0}^{12}{x_i^{j}\cdot w_j} + b
$$
梯度的定义：

$$
𝑔𝑟𝑎𝑑𝑖𝑒𝑛𝑡 = (\frac{\partial{L}}{\partial{w_0}},\frac{\partial{L}}{\partial{w_1}}, ... ,\frac{\partial{L}}{\partial{w_{12}}} ,\frac{\partial{L}}{\partial{b}})
$$
可以计算出$L$对$w$和$b$的偏导数：

$$
\frac{\partial{L}}{\partial{w_j}} = \frac{1}{N}\sum_{i=1}^N{(z_i - y_i)\frac{\partial{z_i}}{\partial{w_j}}} = \frac{1}{N}\sum_{i=1}^N{(z_i - y_i)x_i^{j}} \\

\frac{\partial{L}}{\partial{b}} = \frac{1}{N}\sum_{i=1}^N{(z_i - y_i)\frac{\partial{z_i}}{\partial{b}}} = \frac{1}{N}\sum_{i=1}^N{(z_i - y_i)}
$$
从导数的计算过程可以看出，因子$\frac{1}{2}$被消掉了，这是因为二次函数求导的时候会产生因子2，这也是我们将损失函数改写的原因。

下面我们考虑只有一个样本的情况下，计算梯度：

$$
L= \frac{1}{2}{(y_i - z_i)^2} \\

z_1 = {x_1^{0}\cdot w_0} + {x_1^{1}\cdot w_1} + ... + {x_1^{12}\cdot w_{12}} + b
$$
可以计算出：

$$
L= \frac{1}{2}{({x_1^{0}\cdot w_0} + {x_1^{1}\cdot w_1} + ... + {x_1^{12}\cdot w_{12}} + b - y_1)^2}
$$
可以计算出$L$对$w$和$b$的偏导数：

$$
\frac{\partial{L}}{\partial{w_0}} = ({x_1^{0}\cdot w_0} + {x_1^{1}\cdot w_1} + ... + {x_1^{12}\cdot w_12} + b - y_1)\cdot x_1^{0}=({z_1} - {y_1})\cdot x_1^{0} \\

\frac{\partial{L}}{\partial{b}} = ({x_1^{0}\cdot w_0} + {x_1^{1}\cdot w_1} + ... + {x_1^{12}\cdot w_{12}} + b - y_1)\cdot 1 = ({z_1} - {y_1})
$$
可以通过具体的程序查看每个变量的数据和维度。

```python
x1 = x[0]
y1 = y[0]
z1 = net.forward(x1)
print('x1 {}, shape {}'.format(x1, x1.shape))
print('y1 {}, shape {}'.format(y1, y1.shape))
print('z1 {}, shape {}'.format(z1, z1.shape))
```

```
x1 [0.         0.18       0.07344184 0.         0.31481481 0.57750527
 0.64160659 0.26920314 0.         0.22755741 0.28723404 1.
 0.08967991], shape (13,)
y1 [0.42222222], shape (1,)
z1 [130.86954441], shape (1,)
```

按上面的公式，当只有一个样本时，可以计算某个$w_j$​，比如$w_0$​的梯度。

```python
gradient_w0 = (z1 - y1) * x1[0]
print('gradient_w0 {}'.format(gradient_w0))
```

```
gradient_w0 [0.]
```

同样我们可以计算$w_1$​的梯度。

```python
gradient_w1 = (z1 - y1) * x1[1]
print('gradient_w1 {}'.format(gradient_w1))
```

```
gradient_w1 [23.48051799]
```

依次计算$w_2$​的梯度。

```python
gradient_w2= (z1 - y1) * x1[2]
print('gradient_w1 {}'.format(gradient_w2))
```

```
gradient_w1 [9.58029163]
```

聪明的读者可能已经想到，写一个for循环即可计算从$w_0$​到$w_{12}$​的所有权重的梯度，该方法读者可以自行实现。



##### 使用Numpy进行梯度计算

基于Numpy广播机制（对向量和矩阵计算如同对1个单一变量计算一样），可以更快速的实现梯度计算。计算梯度的代码中直接用$(z_1 - y_1) \cdot x_1$，得到的是一个13维的向量，每个分量分别代表该维度的梯度。

```python
gradient_w = (z1 - y1) * x1
print('gradient_w_by_sample1 {}, gradient.shape {}'.format(gradient_w, gradient_w.shape))
```

```
gradient_w_by_sample1 [  0.          23.48051799   9.58029163   0.          41.06674958
  75.33401592  83.69586171  35.11682862   0.          29.68425495
  37.46891169 130.44732219  11.69850434], gradient.shape (13,)
```

输入数据中有多个样本，每个样本都对梯度有贡献。如上代码计算了只有样本1时的梯度值，同样的计算方法也可以计算样本2和样本3对梯度的贡献。

```python
x2 = x[1]
y2 = y[1]
z2 = net.forward(x2)
gradient_w = (z2 - y2) * x2
print('gradient_w_by_sample2 {}, gradient.shape {}'.format(gradient_w, gradient_w.shape))
gradient_w_by_sample2 [2.54738434e-02 0.00000000e+00 2.83333765e+01 0.00000000e+00
 1.86624242e+01 5.91703008e+01 8.45121992e+01 3.76793284e+01
 4.69458498e+00 1.23980167e+01 5.97311025e+01 1.07975454e+02
 2.20777626e+01], gradient.shape (13,)
x3 = x[2]
y3 = y[2]
z3 = net.forward(x3)
gradient_w = (z3 - y3) * x3
print('gradient_w_by_sample3 {}, gradient.shape {}'.format(gradient_w, gradient_w.shape))
```

```
gradient_w_by_sample3 [3.07963708e-02 0.00000000e+00 3.42860463e+01 0.00000000e+00
 2.25832858e+01 9.07287666e+01 7.83155260e+01 4.55955257e+01
 5.68088867e+00 1.50027645e+01 7.22802431e+01 1.29029688e+02
 8.29246719e+00], gradient.shape (13,)
```

可能有的读者再次想到可以使用for循环把每个样本对梯度的贡献都计算出来，然后再作平均。但是我们不需要这么做，仍然可以使用Numpy的矩阵操作来简化运算，如3个样本的情况。

```python
# 注意这里是一次取出3个样本的数据，不是取出第3个样本
x3samples = x[0:3]
y3samples = y[0:3]
z3samples = net.forward(x3samples)

print('x {}, shape {}'.format(x3samples, x3samples.shape))
print('y {}, shape {}'.format(y3samples, y3samples.shape))
print('z {}, shape {}'.format(z3samples, z3samples.shape))
```

```
x [[0.00000000e+00 1.80000000e-01 7.34418420e-02 0.00000000e+00
  3.14814815e-01 5.77505269e-01 6.41606591e-01 2.69203139e-01
  0.00000000e+00 2.27557411e-01 2.87234043e-01 1.00000000e+00
  8.96799117e-02]
 [2.35922539e-04 0.00000000e+00 2.62405717e-01 0.00000000e+00
  1.72839506e-01 5.47997701e-01 7.82698249e-01 3.48961980e-01
  4.34782609e-02 1.14822547e-01 5.53191489e-01 1.00000000e+00
  2.04470199e-01]
 [2.35697744e-04 0.00000000e+00 2.62405717e-01 0.00000000e+00
  1.72839506e-01 6.94385898e-01 5.99382080e-01 3.48961980e-01
  4.34782609e-02 1.14822547e-01 5.53191489e-01 9.87519166e-01
  6.34657837e-02]], shape (3, 13)
y [[0.42222222]
 [0.36888889]
 [0.66      ]], shape (3, 1)
z [[130.86954441]
 [108.34434338]
 [131.3204395 ]], shape (3, 1)
```

上面的x3samples, y3samples, z3samples的第一维大小均为3，表示有3个样本。下面计算这3个样本对梯度的贡献。

```python
gradient_w = (z3samples - y3samples) * x3samples
print('gradient_w {}, gradient.shape {}'.format(gradient_w, gradient_w.shape))
```

```
gradient_w [[0.00000000e+00 2.34805180e+01 9.58029163e+00 0.00000000e+00
  4.10667496e+01 7.53340159e+01 8.36958617e+01 3.51168286e+01
  0.00000000e+00 2.96842549e+01 3.74689117e+01 1.30447322e+02
  1.16985043e+01]
 [2.54738434e-02 0.00000000e+00 2.83333765e+01 0.00000000e+00
  1.86624242e+01 5.91703008e+01 8.45121992e+01 3.76793284e+01
  4.69458498e+00 1.23980167e+01 5.97311025e+01 1.07975454e+02
  2.20777626e+01]
 [3.07963708e-02 0.00000000e+00 3.42860463e+01 0.00000000e+00
  2.25832858e+01 9.07287666e+01 7.83155260e+01 4.55955257e+01
  5.68088867e+00 1.50027645e+01 7.22802431e+01 1.29029688e+02
  8.29246719e+00]], gradient.shape (3, 13)
```

此处可见，计算梯度`gradient_w`的维度是$3 \times 13$​，并且其第1行与上面第1个样本计算的梯度gradient_w_by_sample1一致，第2行与上面第2个样本计算的梯度gradient_w_by_sample2一致，第3行与上面第3个样本计算的梯度gradient_w_by_sample3一致。这里使用矩阵操作，可以更加方便的对3个样本分别计算各自对梯度的贡献。

那么对于有N个样本的情形，我们可以直接使用如下方式计算出所有样本对梯度的贡献，这就是使用Numpy库广播功能带来的便捷。 小结一下这里使用Numpy库的广播功能：

- 一方面可以扩展参数的维度，代替for循环来计算1个样本对从$w_0$到$w_12$的所有参数的梯度。
- 另一方面可以扩展样本的维度，代替for循环来计算样本0到样本403对参数的梯度。

```python
z = net.forward(x)
gradient_w = (z - y) * x
print('gradient_w shape {}'.format(gradient_w.shape))
print(gradient_w)
```

```
gradient_w shape (404, 13)
[[0.00000000e+00 2.34805180e+01 9.58029163e+00 ... 3.74689117e+01
  1.30447322e+02 1.16985043e+01]
 [2.54738434e-02 0.00000000e+00 2.83333765e+01 ... 5.97311025e+01
  1.07975454e+02 2.20777626e+01]
 [3.07963708e-02 0.00000000e+00 3.42860463e+01 ... 7.22802431e+01
  1.29029688e+02 8.29246719e+00]
 ...
 [3.97706874e+01 0.00000000e+00 1.74130673e+02 ... 2.01043762e+02
  2.48659390e+02 1.27554582e+02]
 [2.69696515e+01 0.00000000e+00 1.75225687e+02 ... 2.02308019e+02
  2.34270491e+02 1.28287658e+02]
 [6.08972123e+01 0.00000000e+00 1.53017134e+02 ... 1.76666981e+02
  2.18509161e+02 1.08772220e+02]]
```

上面gradient_w的每一行代表了一个样本对梯度的贡献。根据梯度的计算公式，总梯度是对每个样本对梯度贡献的平均值。

$$
\frac{\partial{L}}{\partial{w_j}} = \frac{1}{N}\sum_{i=1}^N{(z_i - y_i)\frac{\partial{z_i}}{\partial{w_j}}} = \frac{1}{N}\sum_{i=1}^N{(z_i - y_i)x_i^{j}}
$$
我们也可以使用Numpy的均值函数来完成此过程：

```python
# axis = 0 表示把每一行做相加然后再除以总的行数
gradient_w = np.mean(gradient_w, axis=0)
print('gradient_w ', gradient_w.shape)
print('w ', net.w.shape)
print(gradient_w)
print(net.w)
```

```
gradient_w  (13,)
w  (13, 1)
[  4.6555403   19.35268996  55.88081118  14.00266972  47.98588869
  76.87210821  94.8555119   36.07579608  45.44575958  59.65733292
  83.65114918 134.80387478  38.93998153]
[[ 1.76405235e+00]
 [ 4.00157208e-01]
 [ 9.78737984e-01]
 [ 2.24089320e+00]
 [ 1.86755799e+00]
 [ 1.59000000e+02]
 [ 9.50088418e-01]
 [-1.51357208e-01]
 [-1.03218852e-01]
 [ 1.59000000e+02]
 [ 1.44043571e-01]
 [ 1.45427351e+00]
 [ 7.61037725e-01]]
```

我们使用Numpy的矩阵操作方便地完成了gradient的计算，但引入了一个问题，`gradient_w`的形状是(13,)，而ww*w*的维度是(13, 1)。导致该问题的原因是使用`np.mean`函数时消除了第0维。为了加减乘除等计算方便，`gradient_w`和ww*w*必须保持一致的形状。因此我们将`gradient_w`的维度也设置为(13,1)，代码如下：

```python
gradient_w = gradient_w[:, np.newaxis]
print('gradient_w shape', gradient_w.shape)
```

```
gradient_w shape (13, 1)
```

综合上面的剖析，计算梯度的代码如下所示。

```py
z = net.forward(x)
gradient_w = (z - y) * x
gradient_w = np.mean(gradient_w, axis=0)
gradient_w = gradient_w[:, np.newaxis]
gradient_w
```

```
array([[ 4.6555403 ], [ 19.35268996], [ 55.88081118], [ 14.00266972], [ 47.98588869], [ 76.87210821], [ 94.8555119 ], [ 36.07579608], [ 45.44575958], [ 59.65733292], [ 83.65114918], [134.80387478], [ 38.93998153]])
```

上述代码非常简洁地完成了$w$​的梯度计算。同样，计算$b$​的梯度的代码也是类似的原理。

```python
gradient_b = (z - y)
gradient_b = np.mean(gradient_b)
# 此处b是一个数值，所以可以直接用np.mean得到一个标量
gradient_b
```

```
142.50289323156107
```

将上面计算$w$​和$b$​的梯度的过程，写成Network类的`gradient`函数，实现方法如下所示。

```python
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
        
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost
    
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        
        return gradient_w, gradient_b
# 调用上面定义的gradient函数，计算梯度
# 初始化网络
net = Network(13)
# 设置[w5, w9] = [-100., -100.]
net.w[5] = -100.0
net.w[9] = -100.0

z = net.forward(x)
loss = net.loss(z, y)
gradient_w, gradient_b = net.gradient(x, y)
gradient_w5 = gradient_w[5][0]
gradient_w9 = gradient_w[9][0]
print('point {}, loss {}'.format([net.w[5][0], net.w[9][0]], loss))
print('gradient {}'.format([gradient_w5, gradient_w9]))
```

```
point [-100.0, -100.0], loss 7873.345739941161
gradient [-45.87968288123223, -35.50236884482904]
```



##### 确定损失函数更小的点

下面我们开始研究更新梯度的方法。首先沿着梯度的反方向移动一小步，找到下一个点P1，观察损失函数的变化。

```python
# 在[w5, w9]平面上，沿着梯度的反方向移动到下一个点P1
# 定义移动步长 eta
eta = 0.1
# 更新参数w5和w9
net.w[5] = net.w[5] - eta * gradient_w5
net.w[9] = net.w[9] - eta * gradient_w9
# 重新计算z和loss
z = net.forward(x)
loss = net.loss(z, y)
gradient_w, gradient_b = net.gradient(x, y)
gradient_w5 = gradient_w[5][0]
gradient_w9 = gradient_w[9][0]
print('point {}, loss {}'.format([net.w[5][0], net.w[9][0]], loss))
print('gradient {}'.format([gradient_w5, gradient_w9]))
```

```
point [-95.41203171187678, -96.4497631155171], loss 7214.694816482369
gradient [-43.883932999069096, -34.019273908495926]
```

运行上面的代码，可以发现沿着梯度反方向走一小步，下一个点的损失函数的确减少了。感兴趣的话，大家可以尝试不停的点击上面的代码块，观察损失函数是否一直在变小。

在上述代码中，每次更新参数使用的语句： `net.w[5] = net.w[5] - eta * gradient_w5`

- 相减：参数需要向梯度的反方向移动。
- eta：控制每次参数值沿着梯度反方向变动的大小，即每次移动的步长，又称为学习率。

大家可以思考下，为什么之前我们要做输入特征的归一化，保持尺度一致？这是为了让统一的步长更加合适。

如 **图8** 所示，特征输入归一化后，不同参数输出的Loss是一个比较规整的曲线，学习率可以设置成统一的值 ；特征输入未归一化时，不同特征对应的参数所需的步长不一致，尺度较大的参数需要大步长，尺寸较小的参数需要小步长，导致无法设置统一的学习率。

![img](https://ai-studio-static-online.cdn.bcebos.com/903f552bc55b4a5eba71caa7dd86fd2d7b71b8ebb6cb4500a5f5711f465707f3)


图8：未归一化的特征，会导致不同特征维度的理想步长不同



#### 代码封装Train函数

将上面的循环计算过程封装在`train`和`update`函数中，实现方法如下所示。

```python
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights,1)
        self.w[5] = -100.
        self.w[9] = -100.
        self.b = 0.
        
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost
    
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)        
        return gradient_w, gradient_b
    
    def update(self, gradient_w5, gradient_w9, eta=0.01):
        net.w[5] = net.w[5] - eta * gradient_w5
        net.w[9] = net.w[9] - eta * gradient_w9
        
    def train(self, x, y, iterations=100, eta=0.01):
        points = []
        losses = []
        for i in range(iterations):
            points.append([net.w[5][0], net.w[9][0]])
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            gradient_w5 = gradient_w[5][0]
            gradient_w9 = gradient_w[9][0]
            self.update(gradient_w5, gradient_w9, eta)
            losses.append(L)
            if i % 50 == 0:
                print('iter {}, point {}, loss {}'.format(i, [net.w[5][0], net.w[9][0]], L))
        return points, losses

# 获取数据
train_data, test_data = load_data()
x = train_data[:, :-1]
y = train_data[:, -1:]
# 创建网络
net = Network(13)
num_iterations=2000
# 启动训练
points, losses = net.train(x, y, iterations=num_iterations, eta=0.01)

# 画出损失函数的变化趋势
plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()
```

![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYAAAAD8CAYAAAB+UHOxAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvIxREBQAAIABJREFUeJzt3X10XPV95/H3V8+yZD3ZsixLBtlgICaEJwVMnjYJiTEkjWnzcMjmFJd6121Dd5Nmuy3ZdJdu0uxJ2m2S0gd6nODE9CQhhJLizbIhjiFtEoJBBmPAYCSMjS1sS7b8bMt6+u4f85MZC401Y83MleZ+XufMmXt/85s737mS5qN77+/ONXdHRETipyjqAkREJBoKABGRmFIAiIjElAJARCSmFAAiIjGlABARiSkFgIhITCkARERiSgEgIhJTJVEXcDazZ8/2tra2qMsQEZlWNm3atN/dGyfqN6UDoK2tjY6OjqjLEBGZVsxsZzr9tAtIRCSm0goAM/sjM3vBzJ43s++bWYWZLTCzjWbWZWY/MLOy0Lc8zHeFx9uSlvP50L7NzG7IzVsSEZF0TBgAZtYC/Geg3d3fChQDtwBfBb7u7hcCB4GV4SkrgYOh/euhH2a2ODzvUmAZ8A9mVpzdtyMiIulKdxdQCVBpZiXADGAP8H7ggfD4WuDmML08zBMev97MLLTf5+6n3P1VoAu4ZvJvQUREzsWEAeDu3cD/Bl4j8cF/GNgEHHL3odBtN9ASpluAXeG5Q6H/rOT2cZ4jIiJ5ls4uoHoS/70vAOYBVSR24eSEma0ysw4z6+jt7c3Vy4iIxF46u4A+ALzq7r3uPgg8CLwTqAu7hABage4w3Q3MBwiP1wIHktvHec5p7r7a3dvdvb2xccJhrCIico7SCYDXgCVmNiPsy78e2Ao8Bnws9FkBPBSm14V5wuOPeuK6k+uAW8IooQXAIuDJ7LyNM3UfOsnXfrqNHfuP52LxIiIFIZ1jABtJHMx9GnguPGc18KfA58ysi8Q+/nvCU+4BZoX2zwF3hOW8ANxPIjx+Atzu7sNZfTfBoRMD3PVoFy/tPZKLxYuIFIS0zgR29zuBO8c0b2ecUTzu3g98PMVyvgx8OcMaM9Y4sxyA3qOncv1SIiLTVkGeCTyrqpwiUwCIiJxNQQZAcZExq7qcHgWAiEhKBRkAAI3V5doCEBE5i4INgDk15fQeUwCIiKRSsAHQWF1OzxEFgIhIKoUbADPL2X/sFCMjHnUpIiJTUsEGwJyZ5QyNOIdODkZdiojIlFSwAdA4swKAnqP9EVciIjI1FXAA6GQwEZGzKdgAmKMAEBE5q4INgNEtAJ0MJiIyvoINgKryEmaUFWsLQEQkhYINAEhsBSgARETGV9ABMGdmuUYBiYikUNABoC0AEZHUCjsA9IVwIiIpFXQAzKmp4Ej/EP2DObnwmIjItDZhAJjZxWa2Oel2xMw+a2YNZrbezDrDfX3ob2Z2l5l1mdkWM7sqaVkrQv9OM1uR+lWzo7Fa5wKIiKSSzjWBt7n7Fe5+BXA1cAL4EYlr/W5w90XAhjAPcCOJC74vAlYBdwOYWQOJy0peS+JSkneOhkaunD4bWF8LLSLyJpnuAroeeMXddwLLgbWhfS1wc5heDtzrCU8AdWbWDNwArHf3Pnc/CKwHlk36HZzF6ZPBjmgkkIjIWJkGwC3A98N0k7vvCdN7gaYw3QLsSnrO7tCWqv0MZrbKzDrMrKO3tzfD8s7UXJv4Qri9hxUAIiJjpR0AZlYGfAT44djH3N2BrHzxvruvdvd2d29vbGyc1LIaqsooKy5ij7YARETeJJMtgBuBp919X5jfF3btEO57Qns3MD/pea2hLVV7zpgZc2rK2actABGRN8kkAD7JG7t/ANYBoyN5VgAPJbXfGkYDLQEOh11FjwBLzaw+HPxdGtpyam5NBXu1BSAi8iYl6XQysyrgg8DvJTV/BbjfzFYCO4FPhPaHgZuALhIjhm4DcPc+M/sS8FTo90V375v0O5hAU20FW18/kuuXERGZdtIKAHc/Dswa03aAxKigsX0duD3FctYAazIv89w111Tw6Is9uDtmls+XFhGZ0gr6TGCAubUVnBwc5kj/UNSliIhMKQUfAE01GgoqIjKegg+AuaPnAuhAsIjIGQo/AMIWgIaCioicqeADYE5N4usgtAUgInKmgg+A8pJiZlWVKQBERMYo+ACAxIFg7QISETlTLAJgbm0FexQAIiJniEUANNVUsE+7gEREzhCLAJhbU8GB4wOcGtKlIUVERsUiAEavC9BzRFcGExEZFYsAaNLJYCIibxKLAJgXAuD1QycjrkREZOqIRwDUVQLw+iFtAYiIjIpFAFSVl1A3o5TuQyeiLkVEZMqIRQAAtNRVagtARCRJWgFgZnVm9oCZvWRmL5rZdWbWYGbrzawz3NeHvmZmd5lZl5ltMbOrkpazIvTvNLMVqV8x++bVVdJ9UMcARERGpbsF8DfAT9z9EuBy4EXgDmCDuy8CNoR5SFw8flG4rQLuBjCzBuBO4FrgGuDO0dDIh8QWgAJARGTUhAFgZrXAe4B7ANx9wN0PAcuBtaHbWuDmML0cuNcTngDqzKwZuAFY7+597n4QWA8sy+q7OYuWukqOnhri8MnBfL2kiMiUls4WwAKgF/i2mT1jZt8KF4lvcvc9oc9eoClMtwC7kp6/O7Slaj+Dma0ysw4z6+jt7c3s3ZzFGyOBtBUgIgLpBUAJcBVwt7tfCRznjd09wOkLwXs2CnL31e7e7u7tjY2N2VgkAC31CgARkWTpBMBuYLe7bwzzD5AIhH1h1w7hvic83g3MT3p+a2hL1Z4X8+oSJ4N1KwBERIA0AsDd9wK7zOzi0HQ9sBVYB4yO5FkBPBSm1wG3htFAS4DDYVfRI8BSM6sPB3+Xhra8mF1VTllxkQJARCQoSbPffwK+a2ZlwHbgNhLhcb+ZrQR2Ap8IfR8GbgK6gBOhL+7eZ2ZfAp4K/b7o7n1ZeRdpKCoy5tVVaCioiEiQVgC4+2agfZyHrh+nrwO3p1jOGmBNJgVm0zwNBRUROS02ZwJDYiiodgGJiCTEKgDm1VXSc/QUA0MjUZciIhK5WAVAS10l7ujykCIixC0AwrkAuw7qW0FFRGIVAOc1zABgd5+OA4iIxCoAmmsrKC4yXuvTFoCISKwCoKS4iJa6SgWAiAgxCwBI7AZSAIiIxDAA5jfMYJcCQEQkfgFwXsMMDhwf4NipoahLERGJVCwDANBWgIjEXmwDQMcBRCTuYhsA2gIQkbiLXQDUziilpqJEWwAiEnuxCwCA82ZpKKiISDwDQOcCiIikFwBmtsPMnjOzzWbWEdoazGy9mXWG+/rQbmZ2l5l1mdkWM7sqaTkrQv9OM1uR6vVybX7DDHb3nWRkJCvXsRcRmZYy2QJ4n7tf4e6jVwa7A9jg7ouADWEe4EZgUbitAu6GRGAAdwLXAtcAd46GRr6d1zCDgeER9h3V10KLSHxNZhfQcmBtmF4L3JzUfq8nPAHUmVkzcAOw3t373P0gsB5YNonXP2ejI4F27NduIBGJr3QDwIGfmtkmM1sV2prcfU+Y3gs0hekWYFfSc3eHtlTtZzCzVWbWYWYdvb29aZaXmQWzqwDYceB4TpYvIjIdpHVReOBd7t5tZnOA9Wb2UvKD7u5mlpUd6u6+GlgN0N7enpOd9PNqKykrKeLV/QoAEYmvtLYA3L073PcAPyKxD39f2LVDuO8J3buB+UlPbw1tqdrzrqjIWDCriu29CgARia8JA8DMqsxs5ug0sBR4HlgHjI7kWQE8FKbXAbeG0UBLgMNhV9EjwFIzqw8Hf5eGtkgsmF3Fq/uPRfXyIiKRS2cXUBPwIzMb7f89d/+JmT0F3G9mK4GdwCdC/4eBm4Au4ARwG4C795nZl4CnQr8vuntf1t5JhhY0VrHhpX0MDY9QUhzL0yFEJOYmDAB33w5cPk77AeD6cdoduD3FstYAazIvM/sWzK5icNjpPnSS82dVRV2OiEjexfZf34VhJNB2HQgWkZiKbQCMDgV9VQeCRSSmYhsADVVl1FSUaCioiMRWbAPAzFjQWK0AEJHYim0AQOI4gAJAROIq1gGwYHYV3YdO0j84HHUpIiJ5F/sAALQVICKxFOsAuHBONQCdPTojWETiJ9YBsLCxiiKDrn1Hoy5FRCTvYh0A5SXFtM2q4uV92gIQkfiJdQAALGqqprNHWwAiEj8KgDkz2XHgBKeGNBJIROJFAdBUzfCI6/KQIhI7CoA5MwF4WQeCRSRmYh8AoyOBNBRUROIm9gFQUVrM+bOq6NKBYBGJmbQDwMyKzewZM/txmF9gZhvNrMvMfmBmZaG9PMx3hcfbkpbx+dC+zcxuyPabOVeL5lRrKKiIxE4mWwCfAV5Mmv8q8HV3vxA4CKwM7SuBg6H966EfZrYYuAW4FFgG/IOZFU+u/OxY1FTNjv3HGRgaiboUEZG8SSsAzKwV+BDwrTBvwPuBB0KXtcDNYXp5mCc8fn3ovxy4z91PufurJK4ZfE023sRkXdQ0k6ER13cCiUispLsF8A3gT4DRf5FnAYfcfSjM7wZawnQLsAsgPH449D/dPs5zInXJ3BoAXtxzJOJKRETyZ8IAMLMPAz3uvikP9WBmq8ysw8w6ent78/GSLGysoqykSAEgIrGSzhbAO4GPmNkO4D4Su37+Bqgzs5LQpxXoDtPdwHyA8HgtcCC5fZznnObuq9293d3bGxsbM35D56K0uIiLmqrZqgAQkRiZMADc/fPu3urubSQO4j7q7p8CHgM+FrqtAB4K0+vCPOHxR93dQ/stYZTQAmAR8GTW3skkLW6uYevrR0iUKiJS+CZzHsCfAp8zsy4S+/jvCe33ALNC++eAOwDc/QXgfmAr8BPgdnefMl/A85bmGg4cH6D36KmoSxERyYuSibu8wd1/Dvw8TG9nnFE87t4PfDzF878MfDnTIvNhcXPiQPDWPUeYU1MRcTUiIrkX+zOBR71l3hsBICISBwqAoKailNb6Sl7co6+EEJF4UAAkSRwIPhx1GSIieaEASPKW5hpe3X+ckwNT5ti0iEjOKACSXDqvhhHXcQARiQcFQJK3tdYBsGX3oYgrERHJPQVAkrm1FcyZWc6W3ToOICKFTwEwxtta63hWWwAiEgMKgDEub61le+9xjvQPRl2KiEhOKQDGeNv8xHGA57UbSEQKnAJgjMtbawF4VgEgIgVOATBG3Ywyzp81QyOBRKTgKQDG8bbWOp7dpQAQkcKmABjH5a21vH64n56j/VGXIiKSMwqAcVx5Xj0AT+/UVoCIFC4FwDje2lJDWUkRHTv6oi5FRCRn0rkofIWZPWlmz5rZC2b2P0P7AjPbaGZdZvYDMysL7eVhvis83pa0rM+H9m1mdkOu3tRklZcUc3lrLR07D0ZdiohIzqSzBXAKeL+7Xw5cASwzsyXAV4Gvu/uFwEFgZei/EjgY2r8e+mFmi0lcU/hSYBnwD2ZWnM03k03tbQ08331Y3wwqIgUrnYvCu7sfC7Ol4ebA+4EHQvta4OYwvTzMEx6/3swstN/n7qfc/VWgi3EuKTlVtJ9fz9CI62shRKRgpXUMwMyKzWwz0AOsB14BDrn7UOiyG2gJ0y3ALoDw+GESF40/3T7Oc6acq89PHAjWcQARKVRpBYC7D7v7FUArif/aL8lVQWa2ysw6zKyjt7c3Vy8zoboZZSyaU63jACJSsDIaBeTuh4DHgOuAOjMrCQ+1At1huhuYDxAerwUOJLeP85zk11jt7u3u3t7Y2JhJeVnX3tbApp0HGRnxSOsQEcmFdEYBNZpZXZiuBD4IvEgiCD4Wuq0AHgrT68I84fFH3d1D+y1hlNACYBHwZLbeSC68va2eo/1DvLhXVwgTkcJTMnEXmoG1YcROEXC/u//YzLYC95nZXwDPAPeE/vcA/2RmXUAfiZE/uPsLZnY/sBUYAm539yk9xOa6C2YB8OtXDnDpvNqIqxERya4JA8DdtwBXjtO+nXFG8bh7P/DxFMv6MvDlzMuMRnNtJQtnV/H4Kwf4D+9eGHU5IiJZpTOBJ/COC2excfsBBodHoi5FRCSrFAATeMcFszk+MKzrBItIwVEATGDJwtHjAPsjrkREJLsUABNoqCpjcXMNv+o6EHUpIiJZpQBIwzsumMWm1w7SPzilBy2JiGREAZCGd1/UyMDQCL/erq0AESkcCoA0XLuggYrSIn7+Uk/UpYiIZI0CIA0VpcW884LZPLqth8RJzSIi058CIE3vu2QOu/pO8krv8ahLERHJCgVAmt53yRwAHtNuIBEpEAqANLXUVXJx00we26YAEJHCoADIwHsvaeTJV/s42j8YdSkiIpOmAMjA9Zc0MTTiPLYtugvViIhkiwIgA1efX0/jzHJ+8vyeqEsREZk0BUAGiouMZZfO5bGXejkxMDTxE0REpjAFQIZuvGwuJweH+VftBhKRaS6dS0LON7PHzGyrmb1gZp8J7Q1mtt7MOsN9fWg3M7vLzLrMbIuZXZW0rBWhf6eZrUj1mlPZNW0NzKoq4/8+p91AIjK9pbMFMAT8F3dfDCwBbjezxcAdwAZ3XwRsCPMAN5K43u8iYBVwNyQCA7gTuJbElcTuHA2N6aSkuIill87l0Zd69OVwIjKtTRgA7r7H3Z8O00dJXBC+BVgOrA3d1gI3h+nlwL2e8ARQZ2bNwA3Aenfvc/eDwHpgWVbfTZ7cdNlcTgwM83OdEyAi01hGxwDMrI3E9YE3Ak3uProfZC/QFKZbgF1JT9sd2lK1TzvXLZzF7OpyHny6O+pSRETOWdoBYGbVwD8Dn3X3I8mPeeIb0rLyLWlmtsrMOsyso7d3ah5oLSku4uYr5vHYth76jg9EXY6IyDlJKwDMrJTEh/933f3B0Lwv7Noh3I/uD+kG5ic9vTW0pWo/g7uvdvd2d29vbGzM5L3k1UevbmVw2Fm3WVsBIjI9pTMKyIB7gBfd/WtJD60DRkfyrAAeSmq/NYwGWgIcDruKHgGWmll9OPi7NLRNS29prmFxcw0PPqMAEJHpKZ0tgHcCvw2838w2h9tNwFeAD5pZJ/CBMA/wMLAd6AK+CXwawN37gC8BT4XbF0PbtPXRq1vZsvswnfuORl2KiEjGbCpf4KS9vd07OjqiLiOl/cdOseR/beB33tHGn314cdTliIgAYGab3L19on46E3gSZleXc8Nb5/LDTbs5OaBzAkRkelEATNKtS87n8MlB/s+zr0ddiohIRhQAk3TNggYuaqrm3id26HrBIjKtKAAmycz47evaeL77CM/sOhR1OSIiaVMAZMFvXtlCdXkJ3/nVjqhLERFJmwIgC6rLS/j3157Hj7e8zmsHTkRdjohIWhQAWbLyXQsoKSpi9S9eiboUEZG0KACypKmmgo9e3cL9HbvpOdofdTkiIhNSAGTR773nAoaGR1jzyx1RlyIiMiEFQBa1za7iw2+bx9rHd9B79FTU5YiInJUCIMs+98GLGBwe4e8e7Yy6FBGRs1IAZFnb7Co+8fb5fO/J1zQiSESmNAVADnzm+kUUmfG19duiLkVEJCUFQA401VSw8l0L+JfNr7Np57T+xmsRKWAKgBy5/X0X0lxbwX//lxcYHtF3BInI1KMAyJGq8hK+8KG3sHXPEb67cWfU5YiIvEk6l4RcY2Y9ZvZ8UluDma03s85wXx/azczuMrMuM9tiZlclPWdF6N9pZivGe61C86HLmnnnhbP4q0e2sfewTg4TkaklnS2A7wDLxrTdAWxw90XAhjAPcCOwKNxWAXdDIjCAO4FrgWuAO0dDo5CZGX9x82UMDo9wx4Nb9HXRIjKlTBgA7v5vwNgjmcuBtWF6LXBzUvu9nvAEUGdmzcANwHp373P3g8B63hwqBWnB7CruWHYJP9/Wy/0du6IuR0TktHM9BtDk7nvC9F6gKUy3AMmfcrtDW6r2WLj1ujauWziLL/34RXb16dwAEZkaJn0Q2BP7NbK2b8PMVplZh5l19Pb2ZmuxkSoqMv7yY2/DDD793afpH9T1g0UkeucaAPvCrh3CfU9o7wbmJ/VrDW2p2t/E3Ve7e7u7tzc2Np5jeVPP/IYZ/PXHL+e57sN88cdboy5HROScA2AdMDqSZwXwUFL7rWE00BLgcNhV9Aiw1Mzqw8HfpaEtVpZeOpff/3cX8L2Nr+l4gIhErmSiDmb2feC9wGwz201iNM9XgPvNbCWwE/hE6P4wcBPQBZwAbgNw9z4z+xLwVOj3RXeP5Smyf7z0Ip7rPsR/e/A5WuoqeeeFs6MuSURiyqby0MT29nbv6OiIuoysO9I/yMfv/jWvHzrJD//gOi6ZWxN1SSJSQMxsk7u3T9RPZwJHoKailG/f9nZmlBezYs2TbO89FnVJIhJDCoCIzKur5N7fvZahYeeT33xCISAieacAiNDFc2fyvf+4hKFh55bVT/DyvqNRlyQiMaIAiNhoCDjw0bsf5/FX9kddkojEhAJgCrh47kx+9Ol3MLemghVrnuSHGiIqInmgAJgiWutn8MAfvIO3tzXwXx/Ywh3/vEVnDItITikAppDaylLu/d1r+PR7L+C+p3Zx89//im17dVxARHJDATDFlBQX8SfLLuHbt72dnqOn+PDf/oJv/OxlBoZGoi5NRAqMAmCKet/Fc1j/R+/hxrc2842fdfIbf/tLftFZGF+OJyJTgwJgCptVXc5dn7ySb93azvGBIX77nif5nW8/qd1CIpIV+iqIaeLU0DD3Pr6Tv320kyP9Qyxd3MSn33chV8yvi7o0EZli0v0qCAXANHPw+ADf/tWrfOfxHRzpH2LJwgY+de35LL20ifKS4qjLE5EpQAFQ4I6dGuJ7G3ey9vGddB86Sf2MUn7rqlZ+4/J5XN5ai5lFXaKIREQBEBPDI84vu/Zz35OvsX7rPoZGnJa6Sm5861w+sLiJq86rp6xEh3pE4kQBEEOHTgywfus+fvL8Xn7RuZ+B4REqS4u5dmED77pwNm9va+AtzTUKBJECpwCIuaP9g/z6lQP8sms/v+zaz/be4wCUlRRx6bwaLm+t47KWWhY1VXPhnGpmlE14bSARmSambACY2TLgb4Bi4Fvu/pVUfRUA2bPn8Emeee0Qm3clbs/tPszJpK+aaK2vZNGcatpmV9FaP4PW+spwm0FtZWmElYtIptINgLz+22dmxcDfAx8EdgNPmdk6d9dV0nOsubaS5ssquemyZgCGhkfY2XeCzn3H6Nx3lM6eY7y87ygbX+3jxMCZ30E0s7yExpnlzKouY3Z1ObOr35huqCpjZkUJMytKqQn3MytKqCjViCSRqS7f2/3XAF3uvh3AzO4DlgMKgDwrKS7igsZqLmisZtlb555ud3cOnRhk98GT7D54gt0HT9J96CS9x05x4NgpOnuO8evtBzh0YvCsyy8rKaKmooSq8hIqS4spLy2msrSIitJiKkqKqSwrpmJ0PrSVFBulxUZJUVHivriIkiKjtLiIkjHtpUWJ++Iio7jIKDIoMsOMMJ9oM0tMF4fHipL6jvYZnbYiKB6dDoOoTt9jp+dHx1eZWdI0Gnkl006+A6AFSP6u493AtXmuQc7CzKivKqO+qozLWmtT9hscHqHv+AB9xwc42j/E0f5BjvYPcSTp/sjJIY6fGqJ/cJj+oRH6B4bpOz7AyYFh+oeG6R9MtPUPDTM4PHWPRZ2rs4YFp5Pl9F1y2NiZD5/xfMYud5y+b6olRX3jtKZ8L+e6TJv0MtMP1nGXmWZNqV4m3ZoyWJ1pLfO9FzXyZx9ePP4CsmTKHfkzs1XAKoDzzjsv4mokldLiIppqKmiqqcjK8oZHnMHhEYZGnKHhEQaHnaGREYaG32gfHE7MD42Ex8O0O4y4MzzijHhiK2YktJ2+jSTmT/f1pL4jY/u/0TeZJ7U5JE2f2R46n552T/R5Y/qN9tHnc8Zyfczj478WyX3HLPOMunlz4/j9xjf+YcI0l5lioZOpKZNlptlEqmOh6b7+ZJc5XmNzXeW4z8+mfAdANzA/ab41tJ3m7quB1ZA4CJy/0iRKiV05Om4gkk/5HhD+FLDIzBaYWRlwC7AuzzWIiAh53gJw9yEz+0PgERLDQNe4+wv5rEFERBLyfgzA3R8GHs7364qIyJn0nQAiIjGlABARiSkFgIhITCkARERiSgEgIhJTU/rroM2sF9g5iUXMBvZnqZxsUl2ZUV2ZUV2ZKcS6znf3xok6TekAmCwz60jnK1HzTXVlRnVlRnVlJs51aReQiEhMKQBERGKq0ANgddQFpKC6MqO6MqO6MhPbugr6GICIiKRW6FsAIiKSQkEGgJktM7NtZtZlZnfk+bXnm9ljZrbVzF4ws8+E9j83s24z2xxuNyU95/Oh1m1mdkMOa9thZs+F1+8IbQ1mtt7MOsN9fWg3M7sr1LXFzK7KUU0XJ62TzWZ2xMw+G8X6MrM1ZtZjZs8ntWW8fsxsRejfaWYrclTXX5nZS+G1f2RmdaG9zcxOJq23f0x6ztXh598Vap/UNSxT1JXxzy3bf68p6vpBUk07zGxzaM/n+kr12RDd71jiKkeFcyPxNdOvAAuBMuBZYHEeX78ZuCpMzwReBhYDfw788Tj9F4cay4EFofbiHNW2A5g9pu0vgTvC9B3AV8P0TcD/I3H1uiXAxjz97PYC50exvoD3AFcBz5/r+gEagO3hvj5M1+egrqVASZj+alJdbcn9xiznyVCrhdpvzEFdGf3ccvH3Ol5dYx7/a+B/RLC+Un02RPY7VohbAKcvPO/uA8Dohefzwt33uPvTYfoo8CKJayGnshy4z91PufurQBeJ95Avy4G1YXotcHNS+72e8ARQZ2bNOa7leuAVdz/byX85W1/u/m9A3zivl8n6uQFY7+597n4QWA8sy3Zd7v5Tdx8Ks0+QuLpeSqG2Gnd/whOfIvcmvZes1XUWqX5uWf97PVtd4b/4TwDfP9sycrS+Un02RPY7VogBMN6F58/2AZwzZtYGXAlsDE1/GDbl1oxu5pHfeh34qZltssS1lwGa3H1PmN4LNEVQ16hbOPMPM+r1BZmvnyjW2++S+E9x1AIze8bM/tXM3h3aWkIt+agrk59bvtfXu4F97t6Z1Jb39TXmsyGy37FCDIApwcwKwUOnAAACaElEQVSqgX8GPuvuR4C7gQuAK4A9JDZD8+1d7n4VcCNwu5m9J/nB8J9OJMPCLHGJ0I8APwxNU2F9nSHK9ZOKmX0BGAK+G5r2AOe5+5XA54DvmVlNHkuacj+3MT7Jmf9k5H19jfPZcFq+f8cKMQAmvPB8rplZKYkf8Hfd/UEAd9/n7sPuPgJ8kzd2W+StXnfvDvc9wI9CDftGd+2E+5581xXcCDzt7vtCjZGvryDT9ZO3+szsd4APA58KHxyEXSwHwvQmEvvXLwo1JO8mykld5/Bzy+f6KgF+C/hBUr15XV/jfTYQ4e9YIQZApBeeD/sY7wFedPevJbUn7z//TWB0hMI64BYzKzezBcAiEgefsl1XlZnNHJ0mcRDx+fD6o6MIVgAPJdV1axiJsAQ4nLSZmgtn/GcW9fpKkun6eQRYamb1YffH0tCWVWa2DPgT4CPufiKpvdHMisP0QhLrZ3uo7YiZLQm/o7cmvZds1pXpzy2ff68fAF5y99O7dvK5vlJ9NhDl79hkjmpP1RuJo+cvk0jzL+T5td9FYhNuC7A53G4C/gl4LrSvA5qTnvOFUOs2JjnS4Cx1LSQxwuJZ4IXR9QLMAjYAncDPgIbQbsDfh7qeA9pzuM6qgANAbVJb3tcXiQDaAwyS2K+68lzWD4l98l3hdluO6uoisR949HfsH0Pfj4af72bgaeA3kpbTTuID+RXg7wgngma5rox/btn+ex2vrtD+HeD3x/TN5/pK9dkQ2e+YzgQWEYmpQtwFJCIiaVAAiIjElAJARCSmFAAiIjGlABARiSkFgIhITCkARERiSgEgIhJT/x8gg9FOjzGARQAAAABJRU5ErkJggg==)

```python
iter 0, point [-99.54120317118768, -99.64497631155172], loss 7873.345739941161
iter 50, point [-78.9761810944732, -83.65939206734069], loss 5131.480704109405
iter 100, point [-62.4493631356931, -70.67918223434114], loss 3346.754494352463
iter 150, point [-49.17799206644332, -60.12620415441553], loss 2184.906016270654
iter 200, point [-38.53070194231174, -51.533984751788346], loss 1428.4172504483342
iter 250, point [-29.998249130283174, -44.52613603923428], loss 935.7392894242679
iter 300, point [-23.169901624519575, -38.79894318028118], loss 614.7592258739251
iter 350, point [-17.71439280083778, -34.10731848231335], loss 405.53408184471505
iter 400, point [-13.364557220746388, -30.253470630210863], loss 269.0551396220099
iter 450, point [-9.904936677384967, -27.077764259976597], loss 179.9364750604248
iter 500, point [-7.161782280775628, -24.451346444229817], loss 121.65711285489998
iter 550, point [-4.994989383373879, -22.270198517465555], loss 83.46491706360901
iter 600, point [-3.2915916915280783, -20.450337700789422], loss 58.36183370758033
iter 650, point [-1.9605131425212885, -18.923946252536773], loss 41.792808952534
iter 700, point [-0.9283343968114077, -17.636248840494844], loss 30.792614998570482
iter 750, point [-0.13587780041668718, -16.542993494033716], loss 23.43065354742935
iter 800, point [0.4645474092373408, -15.60841945615185], loss 18.449664464381506
iter 850, point [0.9113672926170796, -14.803617811655524], loss 15.030615923519784
iter 900, point [1.2355357562745004, -14.105208963393421], loss 12.639705730905764
iter 950, point [1.4619805189121953, -13.494275706622066], loss 10.928795653764196
iter 1000, point [1.6107694974712377, -12.955502492189021], loss 9.670616807081698
iter 1050, point [1.6980516626374353, -12.476481020835202], loss 8.716602071285436
iter 1100, point [1.7368159644039771, -12.04715001603925], loss 7.969442965176621
iter 1150, point [1.7375034995020395, -11.659343238414994], loss 7.365228465612388
iter 1200, point [1.7085012931271857, -11.306424818680442], loss 6.861819342703047
iter 1250, point [1.6565405824483015, -10.982995030930885], loss 6.431280353078019
iter 1300, point [1.5870180647823104, -10.684652890749808], loss 6.054953198278096
iter 1350, point [1.5042550040699705, -10.407804594738165], loss 5.720248083137862
iter 1400, point [1.4117062100403601, -10.14950894127009], loss 5.418553777303124
iter 1450, point [1.3121285818148223, -9.907352585055445], loss 5.143875665274019
iter 1500, point [1.2077170340724794, -9.67934935975478], loss 4.891947653805328
iter 1550, point [1.1002141124777076, -9.463859017459276], loss 4.659652555766873
iter 1600, point [0.990998385834045, -9.259521632951046], loss 4.444643323159747
iter 1650, point [0.8811557188942747, -9.065204645952335], loss 4.245095084874306
iter 1700, point [0.7715367363576023, -8.87996009965401], loss 4.059542401818773
iter 1750, point [0.662803148565214, -8.702990105791185], loss 3.88677206759292
iter 1800, point [0.5554650931141796, -8.533618947271485], loss 3.7257521401326525
iter 1850, point [0.4499112301277286, -8.371270536496699], loss 3.5755846299900256
iter 1900, point [0.3464329929523944, -8.215450195281456], loss 3.4354736574042524
iter 1950, point [0.24524412503452966, -8.065729922139326], loss 3.3047037451160453
<Figure size 432x288 with 1 Axes>
```



#### 训练扩展到全部参数

为了能给读者直观的感受，上面演示的梯度下降的过程仅包含$w_5$和$w_9$两个参数，但房价预测的完整模型，必须要对所有参数$w$和$b$进行求解。这需要将Network中的`update`和`train`函数进行修改。由于不再限定参与计算的参数（所有参数均参与计算），修改之后的代码反而更加简洁。实现逻辑：“前向计算输出、根据输出和真实值计算Loss、基于Loss和输入计算梯度、根据梯度更新参数值”四个部分反复执行，直到到损失函数最小。具体代码如下所示。

```python
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
        
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost
    
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)        
        return gradient_w, gradient_b
    
    def update(self, gradient_w, gradient_b, eta = 0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
        
    def train(self, x, y, iterations=100, eta=0.01):
        losses = []
        for i in range(iterations):
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, eta)
            losses.append(L)
            if (i+1) % 10 == 0:
                print('iter {}, loss {}'.format(i, L))
        return losses

# 获取数据
train_data, test_data = load_data()
x = train_data[:, :-1]
y = train_data[:, -1:]
# 创建网络
net = Network(13)
num_iterations=1000
# 启动训练
losses = net.train(x,y, iterations=num_iterations, eta=0.01)

# 画出损失函数的变化趋势
plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()
```

![img](data:image/png;base64,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)

```
iter 9, loss 5.143394325795511
iter 19, loss 3.097924194225988
iter 29, loss 2.082241020617026
iter 39, loss 1.5673801618157397
iter 49, loss 1.2966204735077431
iter 59, loss 1.1453399043319765
iter 69, loss 1.0530155717435201
iter 79, loss 0.9902292156463155
iter 89, loss 0.9426576903842504
iter 99, loss 0.9033048096880774
iter 109, loss 0.868732003041364
iter 119, loss 0.837229250968144
iter 129, loss 0.807927474161227
iter 139, loss 0.7803677341465797
iter 149, loss 0.7542920908532763
iter 159, loss 0.7295420168915829
iter 169, loss 0.7060090054240882
iter 179, loss 0.6836105084697767
iter 189, loss 0.6622781710179412
iter 199, loss 0.6419520361168637
iter 209, loss 0.622577651786949
iter 219, loss 0.6041045903195837
iter 229, loss 0.5864856570315078
iter 239, loss 0.5696764374763879
iter 249, loss 0.5536350125932015
iter 259, loss 0.5383217588525027
iter 269, loss 0.5236991929680567
iter 279, loss 0.509731841376165
iter 289, loss 0.4963861247069634
iter 299, loss 0.48363025234390233
iter 309, loss 0.47143412454019784
iter 319, loss 0.45976924072044867
iter 329, loss 0.44860861316590983
iter 339, loss 0.4379266855659793
iter 349, loss 0.4276992560632111
iter 359, loss 0.4179034044959738
iter 369, loss 0.4085174235863553
iter 379, loss 0.39952075384787633
iter 389, loss 0.39089392200622347
iter 399, loss 0.382618482740513
iter 409, loss 0.3746769635645124
iter 419, loss 0.36705281267772816
iter 429, loss 0.35973034962581096
iter 439, loss 0.35269471861856694
iter 449, loss 0.3459318443621334
iter 459, loss 0.33942839026966587
iter 469, loss 0.33317171892221653
iter 479, loss 0.3271498546584252
iter 489, loss 0.3213514481781961
iter 499, loss 0.31576574305173283
iter 509, loss 0.3103825440311682
iter 519, loss 0.30519218706757245
iter 529, loss 0.30018551094136725
iter 539, loss 0.29535383041913843
iter 549, loss 0.29068891085453674
iter 559, loss 0.28618294415539336
iter 569, loss 0.28182852604338504
iter 579, loss 0.27761863453655344
iter 589, loss 0.27354660958874766
iter 599, loss 0.2696061338236152
iter 609, loss 0.26579121430413205
iter 619, loss 0.26209616528184804
iter 629, loss 0.25851559187303397
iter 639, loss 0.25504437461176843
iter 649, loss 0.2516776548326958
iter 659, loss 0.2484108208387405
iter 669, loss 0.24523949481147198
iter 679, loss 0.24215952042409844
iter 689, loss 0.2391669511192288
iter 699, loss 0.2362580390155805
iter 709, loss 0.2334292244097483
iter 719, loss 0.2306771258409729
iter 729, loss 0.22799853068858245
iter 739, loss 0.22539038627340982
iter 749, loss 0.22284979143604464
iter 759, loss 0.22037398856623477
iter 769, loss 0.21796035605914357
iter 779, loss 0.2156064011754777
iter 789, loss 0.21330975328373866
iter 799, loss 0.2110681574640261
iter 809, loss 0.20887946845393043
iter 819, loss 0.20674164491810018
iter 829, loss 0.2046527440240648
iter 839, loss 0.20261091630783168
iter 849, loss 0.20061440081366638
iter 859, loss 0.1986615204933024
iter 869, loss 0.19675067785062839
iter 879, loss 0.19488035081864621
iter 889, loss 0.19304908885621125
iter 899, loss 0.1912555092527351
iter 909, loss 0.1894982936296714
iter 919, loss 0.18777618462820625
iter 929, loss 0.18608798277314595
iter 939, loss 0.18443254350353405
iter 949, loss 0.18280877436103968
iter 959, loss 0.18121563232764165
iter 969, loss 0.17965212130459232
iter 979, loss 0.1781172897250724
iter 989, loss 0.1766102282933619
iter 999, loss 0.17513006784373505
<Figure size 432x288 with 1 Axes>
```



#### 随机梯度下降法（ Stochastic Gradient Descent）

在上述程序中，每次损失函数和梯度计算都是基于数据集中的全量数据。对于波士顿房价预测任务数据集而言，样本数比较少，只有404个。但在实际问题中，数据集往往非常大，如果每次都使用全量数据进行计算，效率非常低，通俗地说就是“杀鸡焉用牛刀”。由于参数每次只沿着梯度反方向更新一点点，因此方向并不需要那么精确。一个合理的解决方案是每次从总的数据集中随机抽取出小部分数据来代表整体，基于这部分数据计算梯度和损失来更新参数，这种方法被称作随机梯度下降法（Stochastic Gradient Descent，SGD），核心概念如下：

- mini-batch：每次迭代时抽取出来的一批数据被称为一个mini-batch。
- batch_size：一个mini-batch所包含的样本数目称为batch_size。
- epoch：当程序迭代的时候，按mini-batch逐渐抽取出样本，当把整个数据集都遍历到了的时候，则完成了一轮训练，也叫一个epoch。启动训练时，可以将训练的轮数num_epochs和batch_size作为参数传入。

下面结合程序介绍具体的实现过程，涉及到数据处理和训练过程两部分代码的修改。



##### **数据处理代码修改**

数据处理需要实现拆分数据批次和样本乱序（为了实现随机抽样的效果）两个功能。

```python
# 获取数据
train_data, test_data = load_data()
train_data.shape
```

```
(404, 14)
```

train_data中一共包含404条数据，如果batch_size=10，即取前0-9号样本作为第一个mini-batch，命名train_data1。

```python
train_data1 = train_data[0:10]
train_data1.shape
```

```
(10, 14)
```

使用train_data1的数据（0-9号样本）计算梯度并更新网络参数。

```python
net = Network(13)
x = train_data1[:, :-1]
y = train_data1[:, -1:]
loss = net.train(x, y, iterations=1, eta=0.01)
loss
```

```
[4.497480200683046]
```

再取出10-19号样本作为第二个mini-batch，计算梯度并更新网络参数。

```python
train_data2 = train_data[10:20]
x = train_data2[:, :-1]
y = train_data2[:, -1:]
loss = net.train(x, y, iterations=1, eta=0.01)
loss
```

```
[5.849682302465982]
```

按此方法不断的取出新的mini-batch，并逐渐更新网络参数。

接下来，将train_data分成大小为batch_size的多个mini_batch，如下代码所示：将train_data分成 $\frac{404}{10} + 1 = 41$ 个 mini_batch，其中前40个mini_batch，每个均含有10个样本，最后一个mini_batch只含有4个样本。

```python
batch_size = 10
n = len(train_data)
mini_batches = [train_data[k:k+batch_size] for k in range(0, n, batch_size)]
print('total number of mini_batches is ', len(mini_batches))
print('first mini_batch shape ', mini_batches[0].shape)
print('last mini_batch shape ', mini_batches[-1].shape)
```

```
total number of mini_batches is  41
first mini_batch shape  (10, 14)
last mini_batch shape  (4, 14)
```

另外，这里是按顺序读取mini_batch，而SGD里面是随机抽取一部分样本代表总体。为了实现随机抽样的效果，我们先将train_data里面的样本顺序随机打乱，然后再抽取mini_batch。随机打乱样本顺序，需要用到`np.random.shuffle`函数，下面先介绍它的用法。

> **说明：**
>
> 通过大量实验发现，模型对最后出现的数据印象更加深刻。训练数据导入后，越接近模型训练结束，最后几个批次数据对模型参数的影响越大。为了避免模型记忆影响训练效果，需要进行样本乱序操作。

```python
# 新建一个array
a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
print('before shuffle', a)
np.random.shuffle(a)
print('after shuffle', a)
```

```
before shuffle [ 1  2  3  4  5  6  7  8  9 10 11 12]
after shuffle [ 7  2 11  3  8  6 12  1  4  5 10  9]
```

多次运行上面的代码，可以发现每次执行shuffle函数后的数字顺序均不同。 上面举的是一个1维数组乱序的案例，我们再观察下2维数组乱序后的效果。

```python
# 新建一个array
a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
a = a.reshape([6, 2])
print('before shuffle\n', a)
np.random.shuffle(a)
print('after shuffle\n', a)
```

```
before shuffle
 [[ 1  2]
 [ 3  4]
 [ 5  6]
 [ 7  8]
 [ 9 10]
 [11 12]]
after shuffle
 [[ 1  2]
 [ 3  4]
 [ 5  6]
 [ 9 10]
 [11 12]
 [ 7  8]]
```

观察运行结果可发现，数组的元素在第0维被随机打乱，但第1维的顺序保持不变。例如数字2仍然紧挨在数字1的后面，数字8仍然紧挨在数字7的后面，而第二维的[3, 4]并不排在[1, 2]的后面。将这部分实现SGD算法的代码集成到Network类中的`train`函数中，最终的完整代码如下。

```python
# 获取数据
train_data, test_data = load_data()

# 打乱样本顺序
np.random.shuffle(train_data)

# 将train_data分成多个mini_batch
batch_size = 10
n = len(train_data)
mini_batches = [train_data[k:k+batch_size] for k in range(0, n, batch_size)]

# 创建网络
net = Network(13)

# 依次使用每个mini_batch的数据
for mini_batch in mini_batches:
    x = mini_batch[:, :-1]
    y = mini_batch[:, -1:]
    loss = net.train(x, y, iterations=1)
```



##### **训练过程代码修改**

将每个随机抽取的mini-batch数据输入到模型中用于参数训练。训练过程的核心是两层循环：

1. 第一层循环，代表样本集合要被训练遍历几次，称为“epoch”，代码如下：

```python
for epoch_id in range(num_epochs):
```

1. 第二层循环，代表每次遍历时，样本集合被拆分成的多个批次，需要全部执行训练，称为“iter (iteration)”，代码如下：

```python
for iter_id,mini_batch in emumerate(mini_batches):
```

在两层循环的内部是经典的四步训练流程：前向计算->计算损失->计算梯度->更新参数，这与大家之前所学是一致的，代码如下：

```python
            x = mini_batch[:, :-1]
            y = mini_batch[:, -1:]
            a = self.forward(x)  #前向计算
            loss = self.loss(a, y)  #计算损失
            gradient_w, gradient_b = self.gradient(x, y)  #计算梯度
            self.update(gradient_w, gradient_b, eta)  #更新参数
```

将两部分改写的代码集成到Network类中的`train`函数中，最终的实现如下。

```python
import numpy as np

class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        #np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
        
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost
    
    def gradient(self, x, y):
        z = self.forward(x)
        N = x.shape[0]
        gradient_w = 1. / N * np.sum((z-y) * x, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = 1. / N * np.sum(z-y)
        return gradient_w, gradient_b
    
    def update(self, gradient_w, gradient_b, eta = 0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
            
                
    def train(self, training_data, num_epochs, batch_size=10, eta=0.01):
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epochs):
            # 在每轮迭代开始之前，将训练数据的顺序随机打乱
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                #print(self.w.shape)
                #print(self.b)
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                                 format(epoch_id, iter_id, loss))
        
        return losses

# 获取数据
train_data, test_data = load_data()

# 创建网络
net = Network(13)
# 启动训练
losses = net.train(train_data, num_epochs=50, batch_size=100, eta=0.1)

# 画出损失函数的变化趋势
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()
```

![img](data:image/png;base64,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)

```
Epoch   0 / iter   0, loss = 1.0281
Epoch   0 / iter   1, loss = 0.5048
Epoch   0 / iter   2, loss = 0.6382
Epoch   0 / iter   3, loss = 0.5168
Epoch   0 / iter   4, loss = 0.1951
Epoch   1 / iter   0, loss = 0.6281
Epoch   1 / iter   1, loss = 0.4611
Epoch   1 / iter   2, loss = 0.4520
Epoch   1 / iter   3, loss = 0.3961
Epoch   1 / iter   4, loss = 0.1381
Epoch   2 / iter   0, loss = 0.5642
Epoch   2 / iter   1, loss = 0.4250
Epoch   2 / iter   2, loss = 0.4480
Epoch   2 / iter   3, loss = 0.3881
Epoch   2 / iter   4, loss = 0.1884
Epoch   3 / iter   0, loss = 0.3921
Epoch   3 / iter   1, loss = 0.5582
Epoch   3 / iter   2, loss = 0.3759
Epoch   3 / iter   3, loss = 0.3849
Epoch   3 / iter   4, loss = 0.1425
Epoch   4 / iter   0, loss = 0.3821
Epoch   4 / iter   1, loss = 0.4382
Epoch   4 / iter   2, loss = 0.3864
Epoch   4 / iter   3, loss = 0.4314
Epoch   4 / iter   4, loss = 0.0471
Epoch   5 / iter   0, loss = 0.4264
Epoch   5 / iter   1, loss = 0.3829
Epoch   5 / iter   2, loss = 0.3179
Epoch   5 / iter   3, loss = 0.4149
Epoch   5 / iter   4, loss = 0.1581
Epoch   6 / iter   0, loss = 0.3148
Epoch   6 / iter   1, loss = 0.3532
Epoch   6 / iter   2, loss = 0.4195
Epoch   6 / iter   3, loss = 0.3272
Epoch   6 / iter   4, loss = 1.2465
Epoch   7 / iter   0, loss = 0.3166
Epoch   7 / iter   1, loss = 0.2810
Epoch   7 / iter   2, loss = 0.4126
Epoch   7 / iter   3, loss = 0.3309
Epoch   7 / iter   4, loss = 0.2255
Epoch   8 / iter   0, loss = 0.2555
Epoch   8 / iter   1, loss = 0.3678
Epoch   8 / iter   2, loss = 0.3342
Epoch   8 / iter   3, loss = 0.3806
Epoch   8 / iter   4, loss = 0.0570
Epoch   9 / iter   0, loss = 0.3532
Epoch   9 / iter   1, loss = 0.3973
Epoch   9 / iter   2, loss = 0.1945
Epoch   9 / iter   3, loss = 0.2839
Epoch   9 / iter   4, loss = 0.1604
Epoch  10 / iter   0, loss = 0.3414
Epoch  10 / iter   1, loss = 0.2774
Epoch  10 / iter   2, loss = 0.3439
Epoch  10 / iter   3, loss = 0.2103
Epoch  10 / iter   4, loss = 0.0959
Epoch  11 / iter   0, loss = 0.3004
Epoch  11 / iter   1, loss = 0.2497
Epoch  11 / iter   2, loss = 0.2827
Epoch  11 / iter   3, loss = 0.2987
Epoch  11 / iter   4, loss = 0.0316
Epoch  12 / iter   0, loss = 0.2509
Epoch  12 / iter   1, loss = 0.2535
Epoch  12 / iter   2, loss = 0.2944
Epoch  12 / iter   3, loss = 0.2889
Epoch  12 / iter   4, loss = 0.0547
Epoch  13 / iter   0, loss = 0.2792
Epoch  13 / iter   1, loss = 0.2137
Epoch  13 / iter   2, loss = 0.2427
Epoch  13 / iter   3, loss = 0.2986
Epoch  13 / iter   4, loss = 0.3861
Epoch  14 / iter   0, loss = 0.3261
Epoch  14 / iter   1, loss = 0.2123
Epoch  14 / iter   2, loss = 0.1837
Epoch  14 / iter   3, loss = 0.2968
Epoch  14 / iter   4, loss = 0.0620
Epoch  15 / iter   0, loss = 0.2402
Epoch  15 / iter   1, loss = 0.2823
Epoch  15 / iter   2, loss = 0.2574
Epoch  15 / iter   3, loss = 0.1833
Epoch  15 / iter   4, loss = 0.0637
Epoch  16 / iter   0, loss = 0.1889
Epoch  16 / iter   1, loss = 0.1998
Epoch  16 / iter   2, loss = 0.2031
Epoch  16 / iter   3, loss = 0.3219
Epoch  16 / iter   4, loss = 0.1373
Epoch  17 / iter   0, loss = 0.2042
Epoch  17 / iter   1, loss = 0.2070
Epoch  17 / iter   2, loss = 0.2651
Epoch  17 / iter   3, loss = 0.2137
Epoch  17 / iter   4, loss = 0.0138
Epoch  18 / iter   0, loss = 0.1794
Epoch  18 / iter   1, loss = 0.1575
Epoch  18 / iter   2, loss = 0.2554
Epoch  18 / iter   3, loss = 0.2531
Epoch  18 / iter   4, loss = 0.2192
Epoch  19 / iter   0, loss = 0.1779
Epoch  19 / iter   1, loss = 0.2072
Epoch  19 / iter   2, loss = 0.2140
Epoch  19 / iter   3, loss = 0.2513
Epoch  19 / iter   4, loss = 0.0673
Epoch  20 / iter   0, loss = 0.1634
Epoch  20 / iter   1, loss = 0.1887
Epoch  20 / iter   2, loss = 0.2515
Epoch  20 / iter   3, loss = 0.1924
Epoch  20 / iter   4, loss = 0.0926
Epoch  21 / iter   0, loss = 0.1583
Epoch  21 / iter   1, loss = 0.2319
Epoch  21 / iter   2, loss = 0.1550
Epoch  21 / iter   3, loss = 0.2092
Epoch  21 / iter   4, loss = 0.1959
Epoch  22 / iter   0, loss = 0.2414
Epoch  22 / iter   1, loss = 0.1522
Epoch  22 / iter   2, loss = 0.1719
Epoch  22 / iter   3, loss = 0.1829
Epoch  22 / iter   4, loss = 0.2748
Epoch  23 / iter   0, loss = 0.1861
Epoch  23 / iter   1, loss = 0.1830
Epoch  23 / iter   2, loss = 0.1606
Epoch  23 / iter   3, loss = 0.2351
Epoch  23 / iter   4, loss = 0.1479
Epoch  24 / iter   0, loss = 0.1678
Epoch  24 / iter   1, loss = 0.2080
Epoch  24 / iter   2, loss = 0.1471
Epoch  24 / iter   3, loss = 0.1747
Epoch  24 / iter   4, loss = 0.1607
Epoch  25 / iter   0, loss = 0.1162
Epoch  25 / iter   1, loss = 0.2067
Epoch  25 / iter   2, loss = 0.1692
Epoch  25 / iter   3, loss = 0.1757
Epoch  25 / iter   4, loss = 0.0125
Epoch  26 / iter   0, loss = 0.1707
Epoch  26 / iter   1, loss = 0.1898
Epoch  26 / iter   2, loss = 0.1409
Epoch  26 / iter   3, loss = 0.1501
Epoch  26 / iter   4, loss = 0.1002
Epoch  27 / iter   0, loss = 0.1590
Epoch  27 / iter   1, loss = 0.1801
Epoch  27 / iter   2, loss = 0.1578
Epoch  27 / iter   3, loss = 0.1257
Epoch  27 / iter   4, loss = 0.7750
Epoch  28 / iter   0, loss = 0.1573
Epoch  28 / iter   1, loss = 0.1224
Epoch  28 / iter   2, loss = 0.1353
Epoch  28 / iter   3, loss = 0.1862
Epoch  28 / iter   4, loss = 0.5305
Epoch  29 / iter   0, loss = 0.1981
Epoch  29 / iter   1, loss = 0.1114
Epoch  29 / iter   2, loss = 0.1414
Epoch  29 / iter   3, loss = 0.1856
Epoch  29 / iter   4, loss = 0.0268
Epoch  30 / iter   0, loss = 0.0984
Epoch  30 / iter   1, loss = 0.1528
Epoch  30 / iter   2, loss = 0.1637
Epoch  30 / iter   3, loss = 0.1532
Epoch  30 / iter   4, loss = 0.0846
Epoch  31 / iter   0, loss = 0.1433
Epoch  31 / iter   1, loss = 0.1643
Epoch  31 / iter   2, loss = 0.1202
Epoch  31 / iter   3, loss = 0.1215
Epoch  31 / iter   4, loss = 0.2182
Epoch  32 / iter   0, loss = 0.1567
Epoch  32 / iter   1, loss = 0.1420
Epoch  32 / iter   2, loss = 0.1073
Epoch  32 / iter   3, loss = 0.1496
Epoch  32 / iter   4, loss = 0.0846
Epoch  33 / iter   0, loss = 0.1420
Epoch  33 / iter   1, loss = 0.1369
Epoch  33 / iter   2, loss = 0.0962
Epoch  33 / iter   3, loss = 0.1480
Epoch  33 / iter   4, loss = 0.0687
Epoch  34 / iter   0, loss = 0.1234
Epoch  34 / iter   1, loss = 0.1028
Epoch  34 / iter   2, loss = 0.1407
Epoch  34 / iter   3, loss = 0.1528
Epoch  34 / iter   4, loss = 0.0390
Epoch  35 / iter   0, loss = 0.1113
Epoch  35 / iter   1, loss = 0.1289
Epoch  35 / iter   2, loss = 0.1733
Epoch  35 / iter   3, loss = 0.0892
Epoch  35 / iter   4, loss = 0.0456
Epoch  36 / iter   0, loss = 0.1358
Epoch  36 / iter   1, loss = 0.0782
Epoch  36 / iter   2, loss = 0.1475
Epoch  36 / iter   3, loss = 0.1294
Epoch  36 / iter   4, loss = 0.0442
Epoch  37 / iter   0, loss = 0.1136
Epoch  37 / iter   1, loss = 0.0954
Epoch  37 / iter   2, loss = 0.1542
Epoch  37 / iter   3, loss = 0.1262
Epoch  37 / iter   4, loss = 0.0452
Epoch  38 / iter   0, loss = 0.1277
Epoch  38 / iter   1, loss = 0.1361
Epoch  38 / iter   2, loss = 0.1103
Epoch  38 / iter   3, loss = 0.0920
Epoch  38 / iter   4, loss = 0.4119
Epoch  39 / iter   0, loss = 0.1054
Epoch  39 / iter   1, loss = 0.1165
Epoch  39 / iter   2, loss = 0.1334
Epoch  39 / iter   3, loss = 0.1240
Epoch  39 / iter   4, loss = 0.0672
Epoch  40 / iter   0, loss = 0.1218
Epoch  40 / iter   1, loss = 0.0982
Epoch  40 / iter   2, loss = 0.1077
Epoch  40 / iter   3, loss = 0.1062
Epoch  40 / iter   4, loss = 0.4781
Epoch  41 / iter   0, loss = 0.1541
Epoch  41 / iter   1, loss = 0.1049
Epoch  41 / iter   2, loss = 0.0979
Epoch  41 / iter   3, loss = 0.1042
Epoch  41 / iter   4, loss = 0.0397
Epoch  42 / iter   0, loss = 0.0996
Epoch  42 / iter   1, loss = 0.1031
Epoch  42 / iter   2, loss = 0.1294
Epoch  42 / iter   3, loss = 0.0980
Epoch  42 / iter   4, loss = 0.1135
Epoch  43 / iter   0, loss = 0.1521
Epoch  43 / iter   1, loss = 0.1088
Epoch  43 / iter   2, loss = 0.1089
Epoch  43 / iter   3, loss = 0.0775
Epoch  43 / iter   4, loss = 0.1444
Epoch  44 / iter   0, loss = 0.0827
Epoch  44 / iter   1, loss = 0.0875
Epoch  44 / iter   2, loss = 0.1428
Epoch  44 / iter   3, loss = 0.1002
Epoch  44 / iter   4, loss = 0.0352
Epoch  45 / iter   0, loss = 0.0917
Epoch  45 / iter   1, loss = 0.1193
Epoch  45 / iter   2, loss = 0.0933
Epoch  45 / iter   3, loss = 0.1044
Epoch  45 / iter   4, loss = 0.0064
Epoch  46 / iter   0, loss = 0.1020
Epoch  46 / iter   1, loss = 0.0913
Epoch  46 / iter   2, loss = 0.0882
Epoch  46 / iter   3, loss = 0.1170
Epoch  46 / iter   4, loss = 0.0330
Epoch  47 / iter   0, loss = 0.0696
Epoch  47 / iter   1, loss = 0.0996
Epoch  47 / iter   2, loss = 0.0948
Epoch  47 / iter   3, loss = 0.1109
Epoch  47 / iter   4, loss = 0.5095
Epoch  48 / iter   0, loss = 0.0929
Epoch  48 / iter   1, loss = 0.1220
Epoch  48 / iter   2, loss = 0.1150
Epoch  48 / iter   3, loss = 0.0917
Epoch  48 / iter   4, loss = 0.0968
Epoch  49 / iter   0, loss = 0.0732
Epoch  49 / iter   1, loss = 0.0808
Epoch  49 / iter   2, loss = 0.0896
Epoch  49 / iter   3, loss = 0.1306
Epoch  49 / iter   4, loss = 0.1896
<Figure size 432x288 with 1 Axes>
```

观察上述Loss的变化，随机梯度下降加快了训练过程，但由于每次仅基于少量样本更新参数和计算损失，所以损失下降曲线会出现震荡。

> **说明：**
>
> 由于房价预测的数据量过少，所以难以感受到随机梯度下降带来的性能提升.



#### 总结

本节我们详细介绍了如何使用Numpy实现梯度下降算法，构建并训练了一个简单的线性模型实现波士顿房价预测，可以总结出，使用神经网络建模房价预测有三个要点：

- 构建网络，初始化参数w和b，定义预测和损失函数的计算方法。
- 随机选择初始点，建立梯度的计算方法和参数更新方式。
- 从总的数据集中抽取部分数据作为一个mini_batch，计算梯度并更新参数，不断迭代直到损失函数几乎不再下降。



#### 任务1

思考题

1. 样本归一化：预测时的样本数据同样也需要归一化，但使用训练样本的均值和极值计算，这是为什么？
2. 当部分参数的梯度计算为0（接近0）时，可能是什么情况？是否意味着完成训练？
3. 随机梯度下降的batchsize设置成多少合适？过小有什么问题？过大有什么问题？提示：过大以整个样本集合为例，过小以单个样本为例来思考。
4. 一次训练使用的配置：5个epoch，1000个样本，batchsize=20，最内层循环执行多少轮？



#### 任务2

**基本知识**

> **1. 求导的链式法则**
>
> 链式法则是微积分中的求导法则，用于求一个复合函数的导数，是在微积分的求导运算中一种常用的方法。复合函数的导数将是构成复合这有限个函数在相应点的导数的乘积，就像锁链一样一环套一环，故称链式法则。如 **图9** 所示，如果求最终输出对内层输入（第一层）的梯度，等于外层梯度（第二层）乘以本层函数的梯度。
>
> ![img](https://ai-studio-static-online.cdn.bcebos.com/2beffa3f3d7c402685671b0825561a91c17216fe8b924f64b9f29a96f45cbc85)
>
> 图9：求导的链式法则
>
> 
>
> **2. 计算图的概念**
>
> （1）为何是反向计算梯度？即梯度是由网络后端向前端计算。当前层的梯度要依据处于网络中后一层的梯度来计算，所以只有先算后一层的梯度才能计算本层的梯度。
>
> （2）案例：购买苹果产生消费的计算图。假设一家商店9折促销苹果，每个的单价100元。计算一个顾客总消费的结构如 **图10** 所示。
>
> ![img](https://ai-studio-static-online.cdn.bcebos.com/46c43ead4fa942f5be87f25538a046ff9456516816274cbcb5f6df3768c0fd34)
>
>
> 图10：购买苹果所产生的消费计算图
>
> - 前向计算过程：以黑色箭头表示，顾客购买了2个苹果，再加上九折的折扣，一共消费100*2*0.9=180元。
> - 后向传播过程：以红色箭头表示，根据链式法则，本层的梯度计算 * 后一层传递过来的梯度，所以需从后向前计算。
>
> 最后一层的输出对自身的求导为1。导数第二层根据 **图11** 所示的乘法求导的公式，分别为0.9*1和200*1。同样的，第三层为100 * 0.9=90，2 * 0.9=1.8。
>
> ![img](https://ai-studio-static-online.cdn.bcebos.com/c251a2c290e946f99ce3a3381396c392b50e5a4243c346509bd91177b7f2da90)
>
> 图11：乘法求导的公式

1. 根据 **图12** 所示的乘法和加法的导数公式，完成 **图13** 购买苹果和橘子的梯度传播的题目。

![img](https://ai-studio-static-online.cdn.bcebos.com/4ce8715f03f9477699707056544b1e6363f78aa09fda411d972878abb6d1d26f)


​		图12：乘法和加法的导数公式

![img](https://ai-studio-static-online.cdn.bcebos.com/2fc6665e10f34f9e863172bb399862319f0914467d72457d9e7328616bdbe6df)


​		图13：购买苹果和橘子产生消费的计算图



2. 挑战题：用代码实现两层的神经网络的梯度传播，中间层的尺寸为13【房价预测案例】（教案当前的版本为一层的神经网络），如 **图14** 所示。

![img](https://ai-studio-static-online.cdn.bcebos.com/580f2553aa4643809006f5a8d3deb2aa8dd4e1aa69d94cf6a35ead5fe7cf469e)

​		图14：两层的神经网络

---



### 快速上手Paddle

https://www.paddlepaddle.org.cn/tutorials/projectdetail/3469365

跟随官方教paddle的使用，在学习过程中推荐跟随其中的所有示例手动操作一遍。

#### 任务1

使用paddle完成深度学习必经的MNIST CNN网络的搭建和训练。

首先自行搭建一个全连接神经网络观察效果，经典的全连接神经网络来包含四层网络：输入层、两个隐含层和输出层，将手写数字识别任务通过全连接神经网络表示，如 **图3** 所示。

![img](https://ai-studio-static-online.cdn.bcebos.com/2173259df0704335b230ec158be0427677b9c77fd42348a28f2f8adf1ac1c706)


图3：手写数字识别任务的全连接神经网络结构

- 输入层：将数据输入给神经网络。在该任务中，输入层的尺度为28×28的像素值。
- 隐含层：增加网络深度和复杂度，隐含层的节点数是可以调整的，节点数越多，神经网络表示能力越强，参数量也会增加。在该任务中，中间的两个隐含层为10×10的结构，通常隐含层会比输入层的尺寸小，以便对关键信息做抽象，激活函数使用常见的Sigmoid函数。
- 输出层：输出网络计算结果，输出层的节点数是固定的。如果是回归问题，节点数量为需要回归的数字数量。如果是分类问题，则是分类标签的数量。在该任务中，模型的输出是回归一个数字，输出层的尺寸为1。

> **说明：**
>
> 隐含层引入非线性激活函数Sigmoid是为了增加神经网络的非线性能力。
>
> 举例来说，如果一个神经网络采用线性变换，有四个输入$x_1 \sim x_4$，一个输出$y$。假设第一层的变换是$z_1=x_1-x_2$和$z_2=x_3+x_4$，第二层的变换是$y=z_1+z_2$，则将两层的变换展开后得到$y=x_1-x_2+x_3+x_4$。也就是说，无论中间累积了多少层线性变换，原始输入和最终输出之间依然是线性关系。

Sigmoid是早期神经网络模型中常见的非线性变换函数，通过如下代码，绘制出Sigmoid的函数曲线。

```python
def sigmoid(x):
    # 直接返回sigmoid函数
    return 1. / (1. + np.exp(-x))
 
# param:起点，终点，间距
x = np.arange(-8, 8, 0.2)
y = sigmoid(x)
plt.plot(x, y)
plt.show()
```

![img](data:image/png;base64,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)

```
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2349: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
  if isinstance(obj, collections.Iterator):
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2366: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
  return list(data) if isinstance(data, collections.MappingView) else data
<Figure size 432x288 with 1 Axes>
```

针对手写数字识别的任务，网络层的设计如下：

- 输入层的尺度为28×28，但批次计算的时候会统一加1个维度（大小为batch size）。
- 中间的两个隐含层为10×10的结构，激活函数使用常见的Sigmoid函数。
- 与房价预测模型一样，模型的输出是回归一个数字，输出层的尺寸设置成1。

使用上述方案完成一个神经网络并训练，观察精确度效果。

#### 任务2

虽然使用经典的全连接神经网络可以提升一定的准确率，但其输入数据的形式导致丢失了图像像素间的空间信息，这影响了网络对图像内容的理解。对于计算机视觉问题，效果最好的模型仍然是卷积神经网络。卷积神经网络针对视觉问题的特点进行了网络结构优化，可以直接处理原始形式的图像数据，保留像素间的空间信息，因此更适合处理视觉问题。

卷积神经网络由多个卷积层和池化层组成，如 **图4** 所示。卷积层负责对输入进行扫描以生成更抽象的特征表示，池化层对这些特征表示进行过滤，保留最关键的特征信息。

![img](https://ai-studio-static-online.cdn.bcebos.com/91f3755dfd47461aa04567e73474a3ca56107402feb841a592ddaa7dfcbc67c2)

通常来说，比较经典全连接神经网络和卷积神经网络的损失变化，可以发现卷积神经网络的损失值下降更快，且最终的损失值更小。

构建一个与上图结构一致的卷积神经网络进行训练，观察结果。

#### 任务3

然后使用成熟的神经网络VGG-16对MNIST进行训练，观察市面上成熟的神经网络的准确率。

---

在之前的方案中，我们复用了房价预测模型的损失函数-均方误差。从预测效果来看，虽然损失不断下降，模型的预测值逐渐逼近真实值，但模型的最终效果不够理想。究其根本，不同的深度学习任务需要有各自适宜的损失函数。我们以房价预测和手写数字识别两个任务为例，详细剖析其中的缘由如下：

1. 房价预测是回归任务，而手写数字识别是分类任务，使用均方误差作为分类任务的损失函数存在逻辑和效果上的缺欠。
2. 房价可以是大于0的任何浮点数，而手写数字识别的输出只可能是0~9之间的10个整数，相当于一种标签。
3. 在房价预测的案例中，由于房价本身是一个连续的实数值，因此以模型输出的数值和真实房价差距作为损失函数（Loss）是符合道理的。但对于分类问题，真实结果是分类标签，而模型输出是实数值，导致以两者相减作为损失不具备物理含义。

那么，什么是分类任务的合理输出呢？分类任务本质上是“某种特征组合下的分类概率”，下面以一个简单案例说明，如 **图2** 所示。

![img](https://ai-studio-static-online.cdn.bcebos.com/c9f479c2960140839b259ca7ab2256a0dcd7a714e76a4edfb5377f1566796460)

图2：观测数据和背后规律之间的关系

在本案例中，医生根据肿瘤大小$x$作为肿瘤性质$y$的参考判断（判断的因素有很多，肿瘤大小只是其中之一），那么我们观测到该模型判断的结果是$x$和$y$的标签（1为恶性，0为良性）。而这个数据背后的规律是不同大小的肿瘤，属于恶性肿瘤的概率。观测数据是真实规律抽样下的结果，分类模型应该拟合这个真实规律，输出属于该分类标签的概率。



#### Softmax函数

如果模型能输出10个标签的概率，对应真实标签的概率输出尽可能接近100%，而其他标签的概率输出尽可能接近0%，且所有输出概率之和为1。这是一种更合理的假设！与此对应，真实的标签值可以转变成一个10维度的one-hot向量，在对应数字的位置上为1，其余位置为0，比如标签“6”可以转变成[0,0,0,0,0,0,1,0,0,0]。

为了实现上述思路，需要引入Softmax函数，它可以将原始输出转变成对应标签的概率，公式如下，其中CC*C*是标签类别个数。
$$
softmax(x_i) = \frac {e^{x_i}}{\sum_{j=0}^N{e^{x_j}}}, i=0, ..., C-1
$$
从公式的形式可见，每个输出的范围均在0~1之间，且所有输出之和等于1，这是这种变换后可被解释成概率的基本前提。对应到代码上，需要在前向计算中，对全连接网络的输出层增加一个Softmax运算，`outputs = F.softmax(outputs)`。

**图3** 是一个三个标签的分类模型（三分类）使用的Softmax输出层，从中可见原始输出的三个数字3、1、-3，经过Softmax层后转变成加和为1的三个概率值0.88、0.12、0。

![img](https://ai-studio-static-online.cdn.bcebos.com/ef129caf64254318821e9410bb71ab1f45fff20e4282482986081d44a1e3bcbb)


图3：网络输出层改为softmax函数

上文解释了为何让分类模型的输出拟合概率的原因，但为何偏偏用Softmax函数完成这个职能？ 下面以二分类问题（只输出两个标签）进行原理的探讨。

对于二分类问题，使用两个输出接入Softmax作为输出层，等价于使用单一输出接入Sigmoid函数。如 **图4** 所示，利用两个标签的输出概率之和为1的条件，Softmax输出0.6和0.4两个标签概率，从数学上等价于输出一个标签的概率0.6。

![img](https://ai-studio-static-online.cdn.bcebos.com/4dbdf378438f42b0bc6de6f11955834b7063cc6916544017b0af2ccf1f730984)

图4：对于二分类问题，等价于单一输出接入Sigmoid函数

在这种情况下，只有一层的模型为$S(w^{T}x_i)$，S为Sigmoid函数。模型预测为1的概率为$S(w^{T}x_i)$，模型预测为0的概率为$1-S(w^{T}x_i)$。

**图5** 是肿瘤大小和肿瘤性质的数据图。从图中可发现，往往尺寸越大的肿瘤几乎全部是恶性，尺寸极小的肿瘤几乎全部是良性。只有在中间区域，肿瘤的恶性概率会从0逐渐到1（绿色区域），这种数据的分布是符合多数现实问题的规律。如果我们直接线性拟合，相当于红色的直线，会发现直线的纵轴0-1的区域会拉的很长，而我们期望拟合曲线0-1的区域与真实的分类边界区域重合。那么，观察下Sigmoid的曲线趋势可以满足我们对个问题的一切期望，它的概率变化会集中在一个边界区域，有助于模型提升边界区域的分辨率。

![img](https://ai-studio-static-online.cdn.bcebos.com/bbf5e0eda62c44bb84528dbfd8642ef901b2dc42c6f541bc8cd0b75b967dc934)

图5：使用Sigmoid拟合输出可提高分类模型对边界的分辨率

这就类似于公共区域使用的带有恒温装置的热水器温度阀门，如 **图6** 所示。由于人体适应的水温在34度-42度之间，我们更期望阀门的水温条件集中在这个区域，而不是在0-100度之间线性分布。

![img](https://ai-studio-static-online.cdn.bcebos.com/9d05d75c9db44d95b8cdec6fe1615e24d9a24c0ce4f64954a2a5659aaaa7437b)

图6：热水器水温控制



#### 交叉熵

在模型输出为分类标签的概率时，直接以标签和概率做比较也不够合理，人们更习惯使用交叉熵误差作为分类问题的损失衡量。

交叉熵损失函数的设计是基于最大似然思想：最大概率得到观察结果的假设是真的。如何理解呢？举个例子来说，如 **图7** 所示。有两个外形相同的盒子，甲盒中有99个白球，1个蓝球；乙盒中有99个蓝球，1个白球。一次试验取出了一个蓝球，请问这个球应该是从哪个盒子中取出的？

![img](https://ai-studio-static-online.cdn.bcebos.com/13a942e5ec7f4e91badb2f4613c6f71a00e51c8afb6a435e94a0b47cedac9515)

图7：体会最大似然的思想

相信大家简单思考后均会得出更可能是从乙盒中取出的，因为从乙盒中取出一个蓝球的概率更高（$P(D|h)$），所以观察到一个蓝球更可能是从乙盒中取出的($P(h|D)$)。$D$是观测的数据，即蓝球白球；$h$是模型，即甲盒乙盒。这就是贝叶斯公式所表达的思想：
$$
P(h|D) ∝ P(h) \cdot P(D|h)
$$
依据贝叶斯公式，某二分类模型“生成”n个训练样本的概率：
$$
P(x_1)\cdot S(w^{T}x_1)\cdot P(x_2)\cdot(1-S(w^{T}x_2))\cdot … \cdot P(x_n)\cdot S(w^{T}x_n)
$$

> **说明：**
>
> 对于二分类问题，模型为$S(w^{T}x_i)$，S为Sigmoid函数。当$y_i=1$，概率为$S(w^{T}x_i)$；当$y_i=0$，概率为$1-S(w^{T}x_i)$。

经过公式推导，使得上述概率最大等价于最小化交叉熵，得到交叉熵的损失函数。交叉熵的公式如下：
$$
L = -[\sum_{k=1}^{n} t_k\log y_k +(1- t_k)\log(1-y_k)]
$$
其中，$⁡\log$表示以$e$为底数的自然对数。$y_k$代表模型输出，$t_k$代表各个标签。$t_k$中只有正确解的标签为1，其余均为0（one-hot表示）。

因此，交叉熵只计算对应着“正确解”标签的输出的自然对数。比如，假设正确标签的索引是“2”，与之对应的神经网络的输出是0.6，则交叉熵误差是$−\log 0.6 = 0.51$；若“2”对应的输出是0.1，则交叉熵误差为$−\log 0.1 = 2.30$。由此可见，交叉熵误差的值是由正确标签所对应的输出结果决定的。

自然对数的函数曲线可由如下代码实现。

```python
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0.01,1,0.01)
y = np.log(x)
plt.title("y=log(x)") 
plt.xlabel("x") 
plt.ylabel("y") 
plt.plot(x,y)
plt.show()
plt.figure()
```

![img](data:image/png;base64,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)

```
<Figure size 432x288 with 1 Axes>
<Figure size 432x288 with 0 Axes>
<Figure size 432x288 with 0 Axes>
```

如自然对数的图形所示，当x等于1时，y为0；随着x向0靠近，y逐渐变小。因此，正确解标签对应的输出越大，交叉熵的值越接近0；当输出为1时，交叉熵误差为0。反之，如果正确解标签对应的输出越小，则交叉熵的值越大。

#### 任务4：损失函数

使用Cross_entropy损失函数对MNIST进行训练，对比softmax的结果。

#### 任务5：学习率

在深度学习神经网络模型中，通常使用标准的随机梯度下降算法更新参数，学习率代表参数更新幅度的大小，即步长。当学习率最优时，模型的有效容量最大，最终能达到的效果最好。学习率和深度学习任务类型有关，合适的学习率往往需要大量的实验和调参经验。探索学习率最优值时需要注意如下两点：

- **学习率不是越小越好**。学习率越小，损失函数的变化速度越慢，意味着我们需要花费更长的时间进行收敛，如 **图2** 左图所示。
- **学习率不是越大越好**。只根据总样本集中的一个批次计算梯度，抽样误差会导致计算出的梯度不是全局最优的方向，且存在波动。在接近最优解时，过大的学习率会导致参数在最优解附近震荡，损失难以收敛，如 **图2** 右图所示。

![img](https://ai-studio-static-online.cdn.bcebos.com/1e0f066dc9fa4e2bbc942447bdc0578c2ffc6afc15684154ae84bcf31b298d7b)


图2: 不同学习率（步长过大/过小）的示意图

在训练前，我们往往不清楚一个特定问题设置成怎样的学习率是合理的，因此在训练时可以尝试调小或调大，通过观察Loss下降的情况判断合理的学习率。

尝试使用不同的学习率观察训练过程的差异。

#### 任务6：主流的学习率优化算法

学习率是优化器的一个参数，调整学习率看似是一件非常麻烦的事情，需要不断的调整步长，观察训练时间和Loss的变化。经过研究员的不断的实验，当前已经形成了四种比较成熟的优化算法：SGD、Momentum、AdaGrad和Adam，效果如 **图3** 所示。

![img](https://ai-studio-static-online.cdn.bcebos.com/f4cf80f95424411a85ad74998433317e721f56ddb4f64e6f8a28a27b6a1baa6b)


图3: 不同学习率算法效果示意图

- **SGD：** 随机梯度下降算法，每次训练少量数据，抽样偏差导致的参数收敛过程中震荡。
- **Momentum：** 引入物理“动量”的概念，累积速度，减少震荡，使参数更新的方向更稳定。

每个批次的数据含有抽样误差，导致梯度更新的方向波动较大。如果我们引入物理动量的概念，给梯度下降的过程加入一定的“惯性”累积，就可以减少更新路径上的震荡，即每次更新的梯度由“历史多次梯度的累积方向”和“当次梯度”加权相加得到。历史多次梯度的累积方向往往是从全局视角更正确的方向，这与“惯性”的物理概念很像，也是为何其起名为“Momentum”的原因。类似不同品牌和材质的篮球有一定的重量差别，街头篮球队中的投手（擅长中远距离投篮）喜欢稍重篮球的比例较高。一个很重要的原因是，重的篮球惯性大，更不容易受到手势的小幅变形或风吹的影响。

- **AdaGrad：** 根据不同参数距离最优解的远近，动态调整学习率。学习率逐渐下降，依据各参数变化大小调整学习率。

通过调整学习率的实验可以发现：当某个参数的现值距离最优解较远时（表现为梯度的绝对值较大），我们期望参数更新的步长大一些，以便更快收敛到最优解。当某个参数的现值距离最优解较近时（表现为梯度的绝对值较小），我们期望参数的更新步长小一些，以便更精细的逼近最优解。类似于打高尔夫球，专业运动员第一杆开球时，通常会大力打一个远球，让球尽量落在洞口附近。当第二杆面对离洞口较近的球时，他会更轻柔而细致的推杆，避免将球打飞。与此类似，参数更新的步长应该随着优化过程逐渐减少，减少的程度与当前梯度的大小有关。根据这个思想编写的优化算法称为“AdaGrad”，Ada是Adaptive的缩写，表示“适应环境而变化”的意思。RMSProp是在AdaGrad基础上的改进，学习率随着梯度变化而适应，解决AdaGrad学习率急剧下降的问题。

- **Adam：** 由于动量和自适应学习率两个优化思路是正交的，因此可以将两个思路结合起来，这就是当前广泛应用的算法。

> **说明：**
>
> 每种优化算法均有更多的参数设置。理论最合理的未必在具体案例中最有效，所以模型调参是很有必要的，最优的模型配置往往是在一定“理论”和“经验”的指导下实验出来的。

尝试选择不同的优化算法训练模型，观察训练时间和损失变化的情况。

---
