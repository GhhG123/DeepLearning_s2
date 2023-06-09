## 深度学习 入门笔记

[TOC]



5/24

在深度学习训练过程中，数据集、模型和代码是密切相关的三个组成部分，它们之间的关系如下：

1. **数据集（Dataset）**：数据集是深度学习的基础，它是一组用于训练和评估模型的样本数据。数据集包含输入数据和相应的标签或目标值。在训练过程中，模型会根据数据集中的样本进行学习和优化。一个好的数据集应该具有代表性、多样性和足够的数量，以便模型能够从中学到泛化能力强的模式和规律。

   [^输入的数据]: 输入数据可以是各种形式，包括文本、图像、音频、视频或其他任意数据类型。它们是算法或模型的输入，用于学习模式、进行预测或执行其他任务。例如，在图像分类任务中，输入数据可能是一组图像，每个图像都表示一个物体、场景或某种视觉信息。在文本情感分析任务中，输入数据可能是一组文本文档或评论。输入数据通常以矩阵或张量的形式表示，其中每个样本占据矩阵中的一行或一个张量中的一个元素。每个样本的特征被编码为矩阵或张量的列或维度。
   [^相应的标签]: "输入数据的对应的标签"是指每个输入数据样本所关联的标签或类别。它表示了给定输入数据应该对应的预期输出。标签通常是离散的值，例如分类任务中的类别标签或回归任务中的数值标签。例如，在图像分类任务中，如果数据集包含猫和狗的图像，那么每个图像样本的标签可能是"猫"或"狗"。在垃圾邮件检测任务中，每个电子邮件样本的标签可能是"垃圾邮件"或"非垃圾邮件"。
   [^目标值]: "输入数据的目标值"是指在监督学习中，模型所尝试预测或分类的目标输出值。目标值与标签具有相同的含义，是模型所要学习的真实值。
   [^]: 在训练模型时，输入数据与其对应的标签或目标值一起被用作训练样本。模型通过观察输入数据和对应的标签来学习数据中的模式和规律，以便在未见过的新数据上进行预测或分类。

   

2. **模型（Model）**：模型是深度学习训练的核心组件，它是由神经网络构成的数学模型，用于对输入数据进行处理和预测。模型的结构和参数决定了其学习和表示能力。在训练过程中，模型会根据输入数据和对应的标签进行反向传播和优化，以最小化预测值与真实值之间的误差。常见的深度学习模型包括卷积神经网络（CNN）、循环神经网络（RNN）和变换器（Transformer）等。

3. **代码（Code）**：代码是实现深度学习训练过程的具体实现。代码包括了数据预处理、模型定义、损失函数定义、优化算法选择、训练循环和评估等步骤。代码负责将数据集输入模型进行训练，通过定义合适的损失函数和优化算法，指导模型参数的更新和优化过程。常见的深度学习框架如TensorFlow和PyTorch提供了丰富的函数和工具，方便开发者编写和运行深度学习训练代码。

   [^1.数据预处理]: ：加载原始数据集：读取原始数据集，可能是图像、文本、音频等。数据清洗：根据具体任务，对数据进行清洗和处理，例如去除噪声、缺失值处理、标准化等。
   [^2.数据转换]: ：将数据转换为适合模型输入的格式，例如将图像转换为张量，将文本转换为词向量等。
   [^3.数据划分]: ：将数据集划分为训练集、验证集和测试集，用于训练、调优和评估模型。
   [^4.模型定义]: ：选择适合任务的深度学习模型架构，如卷积神经网络 (Convolutional Neural Network, CNN)、循环神经网络 (Recurrent Neural Network, RNN)、变换器 (Transformer) 等。定义模型的层结构和参数，包括输入层、隐藏层、输出层等。设置模型的超参数，如学习率、批量大小、激活函数等。
   [^5.损失函数定义]: ：选择适当的损失函数来衡量模型预测与真实目标之间的差异。对于分类任务，常用的损失函数包括交叉熵损失函数。对于回归任务，常用的损失函数包括均方误差损失函数。
   [^6.优化算法选择]: ：选择适合的优化算法来更新模型的参数以最小化损失函数。常用的优化算法包括随机梯度下降 (Stochastic Gradient Descent, SGD)、Adam、Adagrad等。设置优化算法的超参数，如学习率、动量等。
   [^7.训练循环]: ：迭代训练数据集，通过前向传播和反向传播来更新模型的参数。对于每个训练样本，模型根据当前参数进行预测，并计算预测结果与真实标签的损失。使用优化算法根据损失函数的梯度来更新模型的参数。重复上述步骤，直到达到设定的训练轮数或收敛条件。
   [^8.评估]: ：使用验证集或测试集评估训练好的模型在未见过的数据上的性能。计算模型的准确率、精确率、召回率、F1 分数等指标，根据任务需求选择适当的评估指标。根据评估结果进行模型调优或选择最佳模型。
   [^]: 这些步骤通常在代码中以函数、类或模块的形式实现。具体的实现细节可能因使用的深度学习框架而有所不同，如TensorFlow、PyTorch等，但整体流程是类似的。

   

综上所述，数据集提供了训练和评估模型所需的样本数据和标签，模型定义了学习和预测的数学模型，代码则将数据集和模型结合起来，实现了深度学习训练过程中的各种操作和步骤。这三者共同构成了深度学习训练过程中的关键要素。



**checkpoint**

在深度学习中，"checkpoint"是指在训练过程中保存模型的中间状态或参数的文件。它包含了模型的权重、优化器状态、训练轮数等信息。

保存checkpoint的主要目的是在训练过程中定期保存模型的状态，以便在需要时进行恢复或继续训练。当训练过程中出现意外中断或需要在不同的时间点进行模型比较时，checkpoint可以提供一个中间状态，避免从头开始训练。

一般来说，checkpoint文件包含以下内容：

1. 模型的权重：保存了当前训练轮数下模型的参数值。
2. 优化器状态：保存了当前优化算法的状态，包括学习率、动量等信息。
3. 其他训练相关的信息：如当前训练轮数、损失函数值等。

checkpoint文件可以以不同的格式保存，如二进制文件、HDF5文件、TensorFlow的SavedModel格式、PyTorch的.pth文件等，具体取决于所使用的深度学习框架和工具。

保存checkpoint的频率可以根据需要进行调整，通常可以设置为每个训练轮数、每个epoch或根据时间间隔来保存。这样可以在训练过程中间或结束时获得最新的模型状态，并进行模型评估、部署或继续训练等操作。



**ImageNet ** 和 **Tiny-ImageNet**都是广泛用于图像分类任务的数据集，但它们之间存在一些区别。

1. 规模：ImageNet是一个非常大型的数据集，包含超过1000万张图像，涵盖了1000个不同的类别。每个类别都有大约1000张图像。而Tiny-ImageNet是ImageNet的一个子集，规模较小，包含约10万张图像，涵盖了200个类别。
2. 图像分辨率：ImageNet中的图像通常具有较高的分辨率，通常在几百像素到几千像素之间。而Tiny-ImageNet中的图像经过缩放，通常具有较小的分辨率，为64x64或者32x32像素。
3. 类别数量：ImageNet涵盖了大量的类别，从动物、物体到日常物品等各个领域的类别都有覆盖。而Tiny-ImageNet的类别数量较少，主要关注于一些常见的物体和动物类别。
4. 使用场景：ImageNet通常用于训练和评估大型的深度学习模型，特别是卷积神经网络（CNN），以实现高准确率的图像分类任务。Tiny-ImageNet则主要用于研究和教学目的，特别是对于计算资源有限的情况下，可以用来快速验证和验证模型的性能。

总的来说，ImageNet是一个庞大的图像数据集，适用于大规模的图像分类任务和深度学习模型的训练。而Tiny-ImageNet则是一个较小规模的数据集，适用于快速验证模型的性能、教学和研究目的。



- [ ] todo: 深度学习训练过程的图像表示





- [ ] bitahub的演示project成功运行+TensorBoard绘制Graph。
- [ ] *计算图片经过各层处理后的中*间结果的大小。请列出各层的名称及输出的大小。
- [ ] 处理代码：做必要的改动以使得其可以在Tiny-ImageNet 上训练
- [ ] 在代码中增加torch.utils.tensorboard 的代码，以能在TensorBoard 中观察训练集Loss、训练集精度、验证集Loss、验证集精度的变化。
- [ ] 将resnet18 在训练集上的精度（Top5）训练到95%以上，并对第3 步中的曲线进行截图，编写实验报告，分析曲线变化情况。
- [ ] 分别在无GPU、1 个GPU、多个GPU 环境下，**重复上述过程**，观察和量化评价训练速度上的差异，并做讨论。--注意保存2个
- [ ] 至少**保存2 个训练过程中模型的checkpoint，**并使用代码中的--evaluate 选项，对比两次评估的差异，并至少找出其中10 张评判结果不同的图片。
- [ ] **问题**：VScode中先clone再 commit&Sync 和 直接打开repository然后commit&push的区别？为什么速度差这么多？

--新内容在本地文件中