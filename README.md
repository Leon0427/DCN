# DCN 

### deep and cross network

​	DCN是推荐系统常用算法之一，它能够有效地捕获有限度的有效特征的相互作用，学会高度非线性的相互作用，不需要人工特征工程或遍历搜索，并具有较低的计算成本。


​	下面就让我们使用tensorflow从头开始创建一个deep and cross(DCN)吧

### 1.deep and cross network 简要介绍

​	如figure1所示，DCN由
+ embedding and stack layer,
+ cross network
+ deep network
+ combination output layer

    四个部分构成。

    接下来我们就要用tensorflow来实现这四部分网络结构，并用实现的DCN来对数据进行分类了。

![DCN 结构图](./fig/dcn.PNG)

### 2.数据集介绍

​	本样例使用的数据集来自于Kaggle竞赛：Porto Seguro’s Safe Driver Prediction

​	赛题的核心是根据司机的历史数据预测司机次年提出保险赔偿的概率。显然这是一个与ctr预估类似的二分类问题。

​	数据集以csv的格式存放，分为训练集和测试集，可以从.example/data/README.md中的链接中下载

​	数据集每条数据的列数均为59列，其中一些列如“ps_car_14”是连续特征，另外一些列如“ps_car_02_cat”是离散特征。具体的某列特征是离散还是连续可以在.example/config.py中查看

### 3. DCN项目路径介绍

​	本项目路径如下
```
DCN/
  |_____example/
  |           |_____data/             *数据*
  |           |_____config.py         *配置项*
  |           |_____data_reader.py    *数据加载相关*
  |           |_____log.py            *打印日志*
  |           |_____main.py           *训练主流程：包括加载数据、训练DCN*
  |
  |_____fig/
  |
  |_____deep_and_cross.py             *DCN模型，所有第1.节中介绍的4个主要结构都在其中*
```
​	主要的文件有三个：
+ deep_and_cross.py 
+ main.py
+ data_reader.py

### 4. embedding and stacking layer

​	所有第1.节中提到的四个主要结构都在deep_and_cross.py中的_init_graph方法中实现：
```python
    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.feature_index = tf.placeholder(tf.int32, shape=[None, None], name="feature_index")
            self.feature_value = tf.placeholder(tf.float32, shape=[None, None], name="feature_value")
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")
            self.weights = self._initialize_weights()

            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            # 1. embedding layer
            self.embeddings = tf.nn.embedding_lookup(self.weights["embedding_tensor"], self.feature_index)  #
            feature_value = tf.reshape(self.feature_value, shape=[-1, self.field_dim, 1])
            self.embeddings = tf.multiply(self.embeddings, feature_value)  # M * F * K

            # 2. deep network
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_dim * self.embedding_dim])
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])

            for i, layer_wide in enumerate(self.dnn_wides):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" % i)
                self.y_deep = self.dnn_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1 + i])
            # 3. cross network
            input_size = self.field_dim * self.embedding_dim
            self.y_cross_i = tf.reshape(self.embeddings, shape=[-1, 1, input_size])
            self.y_cross = tf.reshape(self.embeddings, shape=[-1, input_size])
            self.y_cross_0 = tf.reshape(self.embeddings, shape=[-1, 1, input_size])
            for i in range(len(self.cross_wides)):
                x0T_x_x1 = tf.reshape(tf.matmul(self.y_cross_0, self.y_cross_i, transpose_a=True),shape=[-1, input_size])
                self.y_cross_i = tf.add(tf.reshape(tf.matmul(x0T_x_x1, self.weights["cross_layer_%d" % i]),shape=[-1,1,input_size]),
                                      self.y_cross_i)
                self.y_cross_i = tf.add(self.y_cross_i, self.weights["cross_bias_%d" % i])
                self.y_cross = tf.concat([self.y_cross, tf.reshape(self.y_cross_i,shape=[-1, input_size])], axis=1)

            # 4. concatenate y_deep and y_cross
            log("concatenating y_deep and y_cross")
            if self.use_deep and self.use_cross:
                concat_input = tf.concat([self.y_cross, self.y_deep], axis=1)
                self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])
            elif self.use_deep:
                concat_input = self.y_deep
                self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])
            elif self.use_cross:
                concat_input = self.y_cross
                self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])
```

​	首先我们单独来看embedding layer的代码段：
```python
            # 1. embedding layer
            self.embeddings = tf.nn.embedding_lookup(self.weights["embedding_tensor"], self.feature_index)  #
            feature_value = tf.reshape(self.feature_value, shape=[-1, self.field_dim, 1])
            self.embeddings = tf.multiply(self.embeddings, feature_value)  # M * F * K
```
> 我们将数据集中的categorical特征one-hot之后的特征总维度设为***M***，将这些特征由1到M进行编号；
> 将one-hot之前的特征列数设为***F***，每一条样本在输入embedding层时由F个处于(1,M)区间的特征编号```self.feature_index```，以及这些特征对应的值```self.feature_value```表示；
> 每个one-hot前的特征对应的embedding宽度为***K***

​	第1行，通过```tf.nn.embedding_lookup```方法来查找、构建原始特征向量对应的embedding向量。```self.weights["embedding_tensor"]```是一个M\*K的矩阵，
我们暂时不考虑batch样本的情况，设输入了一条样本，它的特征编号由```self.feature_index```(1\*F)表示，通过```tf.nn.embedding_lookup```方法，可以获得一个(F*K,)的向量```embedding```。

​	第3行，通过矩阵乘法，将第一步获得的```embedding```乘以特征权值，得到embedding layer的输出。

> 对tensorflow不熟悉的同学，可以自行查找相应方法的api介绍：在ipython中输入```help(tf.nn.embedding_lookup)```即可。



### 4. deep neural network layer(DNN layer)

deep neural network layer由以下代码实现：

```python
            # 2. deep network
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_dim * self.embedding_dim])
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])

            for i, layer_wide in enumerate(self.dnn_wides):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" % i)
                self.y_deep = self.dnn_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1 + i])
```

​	首先，将embedding layer的输出reshape成(?, F*K)形状的二维tensor，这里```?```等于```batch_size```，即批量训练的样本数。

​	然后逐层构建宽度为```dnn_wides[i]```的dense层。

### 5. cross network layer(CN layer)

​	cross network是DCN的精髓：它生成样本的各阶交叉特征。

​	核心公式如下：

​	$$x_{l+1} = x_0x^T_lw_l + b_l + x_l$$

​	代码如下：

```python
        # 3. cross network
        input_size = self.field_dim * self.embedding_dim
        self.y_cross_i = tf.reshape(self.embeddings, shape=[-1, 1, input_size])
        self.y_cross = tf.reshape(self.embeddings, shape=[-1, input_size])
        self.y_cross_0 = tf.reshape(self.embeddings, shape=[-1, 1, input_size])
        for i in range(len(self.cross_wides)):
            x0T_x_x1 = tf.reshape(tf.matmul(self.y_cross_0, self.y_cross_i, transpose_a=True),shape=[-1, input_size])
            self.y_cross_i = tf.add(tf.reshape(tf.matmul(x0T_x_x1, self.weights["cross_layer_%d" % i]),shape=[-1,1,input_size]),
                                  self.y_cross_i)
            self.y_cross_i = tf.add(self.y_cross_i, self.weights["cross_bias_%d" % i])
            self.y_cross = tf.concat([self.y_cross, tf.reshape(self.y_cross_i,shape=[-1, input_size])], axis=1)
```
​	代码中```self.y_cross_0```对应$$x_0$$, 用```self.y_cross_i```对应$$x_l$$, ```self.weights["cross_layer_%d" % i]```对应$$w_l$$, ```self.weights["cross_bias_%d" % i]```对应$$b_l$$

> 本段代码将所有cross层的结果拼接起来作为一个整体的输出，原文中是仅输出最后一层的结果。

### 6. combination output layer

​	该层将DNN layer和CN layer的输出拼接起来，输入一个感知机单元，得到最终的输出。

​	代码如下：

            concat_input = tf.concat([self.y_cross, self.y_deep], axis=1)
            self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])
### 参考资料：

【1】Ruoxi Wang, Bin Fu, Gang Fu, Mingliang Wang,  *Deep & Cross Network for Ad Click Predictions* .

【2】Sklearn 与 TensorFlow 机器学习实用指南，https://hand2st.apachecn.org/#/