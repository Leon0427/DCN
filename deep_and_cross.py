#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/29 11:37
# @Author  : liangxiao
# @Site    : 
# @File    : deep_and_cross.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm


class DeepCrossNetwork(object):
    def __init__(self, field_dim, embedding_dim, cross_deep, dnn_wides,
                 batch_norm=0,
                 l2_reg=0.0,
                 batch_norm_decay=0.995,
                 random_seed=2018,
                 learning_rate=0.001
                 ):
        self.field_dim = field_dim
        self.embedding_dim = embedding_dim
        self.cross_deep = cross_deep
        self.dnn_wides = dnn_wides
        self.batch_norm = batch_norm
        self.l2_reg = l2_reg
        self.batch_norm_decay = batch_norm_decay
        self.random_seed = random_seed
        self.deep_layers_activation = tf.nn.relu
        self.learning_rate = learning_rate

    def _init_graph(self):
        self.graph = tf.graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.feature_index = tf.placeholder(tf.int32, shape=[None, None], name="feature_index")
            self.feature_value = tf.placeholder(tf.float32, shape=[None, None], name="feature_value")
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")
            self.weights = self._initialize_weights()

            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")

            # 1. embedding layer
            self.embeddings = tf.nn.embedding_lookup(self.weights["embedding_tensor"], self.feature_index)  #
            feature_value = tf.reshape(self.feature_value, shape=[-1, self.field_dim, 1])
            self.embeddings = tf.multiply(self.embeddings, feature_value)  # M * F * K

            # 2. cross network
            self.y_cross = tf.reshape(self.embeddings, shape=[-1, self.field_dim * self.embedding_dim])
            self.y_cross_0 = tf.reshape(self.embeddings, shape=[-1, self.field_dim * self.embedding_dim])
            for i in range(self.cross_deep):
                x0_x_x1T = tf.matmul(self.y_cross_0, self.y_cross, transpose_b=True)
                self.y_cross = tf.add(tf.matmul(x0_x_x1T, self.weights["cross_layer_%d"] % i),
                                      self.y_cross)
                self.y_cross = tf.add(self.y_cross, self.weights["cross_bias_%d" % i])

            # 3. deep network
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_dim * self.embedding_dim])
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])

            for i, layer_wide in enumerate(self.dnn_wides):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" % i)
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1 + i])

            # 4. concatenate y_deep and y_cross
            concat_input = tf.concat([self.y_cross,self.y_deep],axis=1)
            self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])

            # 5. loss
            self.out = tf.nn.sigmoid(self.out)
            self.loss = tf.losses.log_loss(self.label, self.out)

            # 6.regularization
            if self.l2_reg >0.0:
                self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["concat_projection"])
                for i in range(len(self.dnn_wides)):
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["layer_%d"%i])

            # 7. optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9,beta2=0.999,
                                                    epsilon=1e-8).minimize(self.loss)

            # 8. init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu":0})

    @staticmethod
    def _initialize_weights():
        weights = dict()
        return weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z


if __name__ == '__main__':
    a = DeepCrossNetwork()
    a.printshit()
