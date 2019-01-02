#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/29 11:37
# @Author  : liangxiao
# @Site    : 
# @File    : deep_and_cross.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from sklearn.metrics import roc_auc_score
import numpy as np
from time import time


class DeepCrossNetwork(object):
    def __init__(self, field_dim, feature_dim, embedding_dim, dnn_wides,dropout_deep, cross_wides,train_phase,
                 batch_norm=0,
                 l2_reg=0.0,
                 batch_norm_decay=0.995,
                 random_seed=2018,
                 learning_rate=0.001,
                 epoch=30,
                 batch_size=1024,
                 verbose = 1,
                 eval_metric = roc_auc_score
                 ):
        self.field_dim = field_dim
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.cross_wides = cross_wides
        self.train_phase = train_phase
        self.dnn_wides = dnn_wides
        self.dropout_deep = dropout_deep
        self.cross_deep = len(cross_wides)
        self.batch_norm = batch_norm
        self.l2_reg = l2_reg
        self.batch_norm_decay = batch_norm_decay
        self.random_seed = random_seed
        self.dnn_activation = tf.nn.relu
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.train_result, self.valid_result = [], []
        self.eval_metric = eval_metric

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
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
            for i in range(self.cross_wides):
                x0_x_x1T = tf.matmul(self.y_cross_0, self.y_cross, transpose_b=True)
                self.y_cross = tf.add(tf.matmul(x0_x_x1T, self.weights["cross_layer_%d" % i]),
                                      self.y_cross)
                self.y_cross = tf.add(self.y_cross, self.weights["cross_bias_%d" % i])

            # 3. deep network
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_dim * self.embedding_dim])
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])

            for i, layer_wide in enumerate(self.dnn_wides):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" % i)
                self.y_deep = self.dnn_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1 + i])

            # 4. concatenate y_deep and y_cross
            concat_input = tf.concat([self.y_cross, self.y_deep], axis=1)
            self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])

            # 5. loss
            self.out = tf.nn.sigmoid(self.out)
            self.loss = tf.losses.log_loss(self.label, self.out)

            # 6.regularization
            if self.l2_reg > 0.0:
                self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["concat_projection"])
                for i in range(len(self.dnn_wides)):
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["layer_%d" % i])

            # 7. optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8).minimize(self.loss)

            # 8. init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def _initialize_weights(self):
        weights = dict()
        # 1. embeddings
        weights["embedding_tensor"] = tf.Variable(tf.random_normal([self.feature_dim, self.embedding_dim], 0.0, 0.1),
                                                  name="embedding_tensor")
        # 2.dnn
        num_layer = len(self.dnn_wides)
        input_size = self.field_dim * self.embedding_dim
        glorot = np.sqrt(2.0 / (input_size + self.dnn_wides[0]))
        weights["layer_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, self.dnn_wides[0])),
                                         dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.dnn_wides[0])),
                                        dtype=np.float32)

        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.dnn_wides[i - 1] + self.dnn_wides[i]))
            weights["layer_%d" % i] = tf.Variable(np.random.normal(loc=0.0, scale=glorot,
                                                                   size=(self.dnn_wides[i - 1], self.dnn_wides[i])),
                                                  dtype=np.float32)
            weights["bias_%d" % i] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(1, self.dnn_wides[i])),
                                                 dtype=np.float32)

        # 3. cross layers
        num_layer = self.cross_deep
        glorot = np.sqrt(2.0 / (input_size + self.cross_wides[0]))
        weights["cross_layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.cross_wides[0])),
            dtype=np.float32)
        weights["cross_bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.cross_wides[0])),
                                              dtype=np.float32)

        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.cross_wides[i - 1] + self.cross_wides[i]))
            weights["cross_layer_%d" % i] = tf.Variable(np.random.normal(loc=0.0, scale=glorot,
                                                                         size=(
                                                                         self.cross_wides[i - 1], self.cross_wides[i])),
                                                        dtype=np.float32)
            weights["cross_bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0.0, scale=glorot, size=(1, self.cross_wides[i])),
                dtype=np.float32)

        # 4. concat layer
        input_size = self.dnn_wides[-1] + self.cross_wides[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(tf.random_normal([input_size, 1], 0.0, glorot), dtype=tf.float32)
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)


        return weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def shuffle_in_unison_scale(self, a, b, c):
        state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(state)
        np.random.shuffle(b)
        np.random.set_state(state)
        np.random.shuffle(c)

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start: end], Xv[start:end], [[y_] for y_ in y[start:end]]

    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {
            self.feature_index:Xi,
            self.feature_value:Xv,
            self.label:y,
            self.dropout_keep_deep:self.dropout_deep,
            self.train_phase:True
        }
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def evaluate(self,Xi, Xv, y):
        y_pred =self.predict(Xi, Xv)
        return self.eval_metric(y, y_pred)

    def predict(self, Xi, Xv):
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while(len(Xi_batch)) > 0:
            num_batch = len(y_batch)
            feed_dict ={
                self.feature_index:Xi_batch,
                self.feature_value:Xv_batch,
                self.label: y_batch,
                self.dropout_keep_deep:[1.0]*len(self.dropout_keep_deep),
                self.train_phase : False
            }
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred =np.reshape(batch_out, (num_batch, ))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch, ))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

            return y_pred

    def train_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                    valid_result[-2] < valid_result[-3] and \
                    valid_result[-3] < valid_result[-4] and \
                    valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                    valid_result[-2] > valid_result[-3] and \
                    valid_result[-3] > valid_result[-4] and \
                    valid_result[-4] > valid_result[-5]:
                    return True
        return False

    def fit(self, Xi_train, Xv_train, y_train, Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scale(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train)/self.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)

            # evaluate training and evaluation dataset
            train_result= self.evaluate(Xi_train, Xv_train, y_train)
            self.train_result.append(train_result)
            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result, valid_result, time() - t1))
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                          % (epoch + 1, train_result, time() - t1))
            if has_valid and early_stopping and self.train_termination(self.valid_result):
                break

        if has_valid and refit:
            if self.greater_is_better:
                best_valid_score = max(self.valid_result)
            else:
                best_valid_score = min(self.valid_result)
            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]
            Xi_train = Xi_train + Xi_valid
            Xv_train = Xv_train + Xv_valid
            y_train = y_train + y_valid
            for epoch in range(100):
                self.shuffle_in_unison_scale(Xi_train, Xv_train, y_train)
                total_batch = int(len(y_train) / self.batch_size)
                for i in range(total_batch):
                    Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                    self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                # check
                train_result = self.evaluate(Xi_train, Xv_train, y_train)
                if abs(train_result - best_valid_score) < 0.001 or \
                    (self.greater_is_better and train_result >best_train_score) or \
                    ((not self.greater_is_better) and train_result < best_train_score):
                    break


if __name__ == '__main__':
    a = DeepCrossNetwork()
