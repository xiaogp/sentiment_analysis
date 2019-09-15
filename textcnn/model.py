# -*- coding: utf-8 -*-


import tensorflow as tf
import tensorflow.contrib.slim as slim


class TextCNN(object):
    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # 定义占位符
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 词嵌入层
        with tf.variable_scope('Embedding'):
            embed = tf.contrib.layers.embed_sequence(self.input_x, vocab_size=vocab_size, embed_dim=embedding_size)
            self.embedded_chars_expanded = tf.expand_dims(embed, -1)

        # 定义多通道卷积 与最大池化网络
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            # 输出num_filters个feature map，每个feature map是sequence_length - filter_size + 1为高，1为宽
            conv = slim.conv2d(self.embedded_chars_expanded, num_outputs=num_filters,
                               kernel_size=[filter_size, embedding_size],
                               stride=1, padding="VALID",
                               activation_fn=tf.nn.leaky_relu, scope="conv%s" % filter_size)  # relu容易全部死掉，leaky_relu默认x<0斜率=0.2
            pooled = slim.max_pool2d(conv, [sequence_length - filter_size + 1, 1], padding='VALID',  # 默认步长=2
                                     scope="pool%s" % filter_size)  # 每个feature map池化之后只有1个值表征特征，一共num_filters个

            pooled_outputs.append(pooled)  # 256×1×1×64，每一个句子样本变成64个卷积核加池化的64个输出值

        # 展开特征，并添加dropout
        num_filters_total = num_filters * len(filter_sizes)  # 渠道书×卷积核数量
        # 256×1×1×192
        self.h_pool = tf.concat(pooled_outputs, 3)  # 对最里面一层合并，每一句话的3，4，5三种维度卷积核结果拼接
        # 改造成256×192的标准的一条样本一个向量的维度
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # 计算L2_loss
        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            # 最后一层全连接softmax
            self.scores = slim.fully_connected(self.h_drop, num_classes, activation_fn=None, scope="fully_connected")
            for tf_var in tf.trainable_variables():
                if ("fully_connected/weights" in tf_var.name):  # 通过变量名判断，只对最后的全连接做l2约束
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))  # 直接取到变量

            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.probs = tf.nn.softmax(logits=self.scores, dim=1, name="probs")

        # 计算交叉熵
        with tf.name_scope("loss"):
            # 全连接加softmax激活函数
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # 计算准确率
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
                
        
    def build_mode(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)  # tuple(梯度，variable的值),minimize分开来写
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # 生成摘要
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)

        # 合并摘要
        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary])

