# DCN

创建日期 2018/12/29

创建人 Leon0427


下面就让我们开始从头创建一个deep and cross(DCN)吧

### 1.deep and cross network 简要介绍

如figure1所示，DCN由
+ embedding and stack layer,
+ cross network
+ deep network
+ combination output layer

四个部分构成。

接下来我们就要用tensorflow来实现这四部分网络结构，并用实现的DCN来对数据进行分类了。

![DCN 结构图](./fig/dcn.PNG)

图来源 Ruoxi Wang, Bin Fu, Gang Fu, Mingliang Wang,  *Deep & Cross Network for Ad Click Predictions* .

### 2.数据集介绍

本样例使用的数据集来自于Kaggle竞赛：Porto Seguro’s Safe Driver Prediction

赛题的核心是根据司机的历史数据预测司机次年提出保险赔偿的概率。显然这是一个与ctr预估类似的二分类问题。

数据集以csv的格式存放，分为训练集和测试集，可以从.example/data/README.md中的链接中下载

数据集每条数据的列数均为59列，其中一些列如“ps_car_14”是连续特征，另外一些列如“ps_car_02_cat”是离散特征。具体的某列特征是离散还是连续可以在.example/config.py中查看

### 3. DCN项目路径介绍

本项目路径如下
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
主要的文件有三个：
+ deep_and_cross.py 
+ main.py
+ data_reader.py

### 4. embedding and stacking layer

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