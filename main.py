# -*- coding: utf-8 -*-

import os
import pickle
import random
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.saved_model import tag_constants

from util import load_yaml_config

config = load_yaml_config("config.yml")
batch_size = config["model"]["batch_size"]
train_epochs = config["model"]["train_epochs"]
embedding_path = config["model"]["embedding_path"]
vocab_path = config["data"]["vocab_path"]
fix_embedding = config["model"]["fix_embedding"]
embedding_size = config["model"]["embedding_size"]
num_layers = config["model"]["num_layers"]
seq_length = config["model"]["seq_length"]
rnn_size = config["model"]["rnn_size"]
label_size = config["model"]["label_size"]
learning_rate = config["model"]["learning_rate"]
use_word2vec = config["model"]["use_word2vec"]
use_bilstm = config["model"]["use_bilstm"]
train_path = config["data"]["train_path"]
summary_path = config["model"]["summary_path"]
test_path = config["data"]["test_path"]
ckpt_path = config["model"]["ckpt_path"]
pb_path = config["model"]["pb_path"]
display_train = config["model"]["display_train"]
display_test = config["model"]["display_test"]


def load_word_embedding(vocab_size, token2index):
    embedding_np = 0.3 * \
                   np.random.randn(vocab_size, embedding_size).astype("float32")

    with open(embedding_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.replace("\n", "").strip().split()
            token = tokens[0]
            vector = tokens[1:]
            if token in token2index:
                embedding_np[token2index[token], :] = vector

    embedding = tf.get_variable("embedding",
                                shape=[vocab_size, embedding_size],
                                initializer=tf.constant_initializer(embedding_np),
                                trainable=fix_embedding)
    return embedding


class Model(object):
    def __init__(self, num_layers, seq_length, embedding_size, vocab_size,
                 rnn_size, label_size, embedding=None, use_bilstm=False):

        self.input_x = tf.placeholder(tf.int64, [None, seq_length], name="input_x")
        self.input_y = tf.placeholder(tf.int64, [None, label_size])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope('embeddingLayer'):
            if embedding:
                embedded = tf.nn.embedding_lookup(embedding, self.input_x)
            else:
                W = tf.get_variable('W', [vocab_size, embedding_size])
                embedded = tf.nn.embedding_lookup(W, self.input_x)

            inputs = tf.split(embedded, seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def get_cell(rnn_size, dropout_keep_prob):
            cell = rnn.LSTMCell(rnn_size, initializer=tf.truncated_normal_initializer(stddev=0.3))
            cell = rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)  # dropout比例
            return cell

        with tf.name_scope('lstm_layer'):
            if use_bilstm:
                cell_fw = rnn.MultiRNNCell([get_cell(rnn_size, self.dropout_keep_prob) for _ in range(num_layers)])
                cell_bw = rnn.MultiRNNCell([get_cell(rnn_size, self.dropout_keep_prob) for _ in range(num_layers)])
                self.outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw=cell_fw,
                                                                  cell_bw=cell_bw,
                                                                  inputs=inputs,
                                                                  dtype=tf.float32)
            else:
                cell = rnn.MultiRNNCell([get_cell(rnn_size, self.dropout_keep_prob) for _ in range(num_layers)])
                self.outputs, _ = rnn.static_rnn(cell, inputs, dtype=tf.float32)

        with tf.name_scope('softmaxLayer'):
            logits = tf.layers.dense(self.outputs[-1], label_size)
            self.probs = tf.nn.softmax(logits, dim=1, name="probs")

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y))
            tf.summary.scalar("loss", self.loss)
        with tf.name_scope('predict'):
            self.correct_pred = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)


def get_batch():
    train_x, train_y = pickle.load(open(train_path, "rb"))
    train_data = list(zip(train_x, train_y))

    for epoch in range(train_epochs):
        random.shuffle(train_data)
        for batch in range(0, len(train_data), batch_size):
            if batch + batch_size < len(train_data):
                yield train_data[batch: (batch + batch_size)]


def train():
    vocab = pickle.load(open(vocab_path, "rb"))
    vocab_size = len(vocab.vocabulary_)
    token2index = vocab.vocabulary_._mapping

    if use_word2vec:
        embedding = load_word_embedding(vocab_size, token2index)

    lstm = Model(num_layers=num_layers,
                 seq_length=seq_length,
                 embedding_size=embedding_size,
                 vocab_size=vocab_size,
                 rnn_size=rnn_size,
                 label_size=label_size,
                 embedding=embedding if use_word2vec else None,
                 use_bilstm=use_bilstm)

    print("{0:-^40}".format("需要训练的参数"))
    for var in tf.trainable_variables():
        print(var.name, var.shape)

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        global_step = tf.Variable(0, name="global_step", trainable=False)  # 设置全局步长
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate). \
            minimize(lstm.loss, global_step=global_step)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        shutil.rmtree(summary_path, ignore_errors=True)
        writer = tf.summary.FileWriter(summary_path, sess.graph)

        def train_step(batch, label):
            feed_dict = {
                lstm.input_x: batch,
                lstm.input_y: label,
                lstm.dropout_keep_prob: 0.7
            }
            _, step, loss, accuracy = sess.run([optimizer, global_step, lstm.loss, lstm.accuracy], feed_dict=feed_dict)

            return step, loss, accuracy

        def dev_step(batch, label):
            feed_dict = {
                lstm.input_x: batch,
                lstm.input_y: label,
                lstm.dropout_keep_prob: 1
            }
            step, loss, accuracy = sess.run([global_step, lstm.loss, lstm.accuracy], feed_dict=feed_dict)
            print("{0:-^40}".format("evaluate"))
            print("global step:{}".format(step), "==>", "loss:%.5f" % loss, "accuracy:%.5f" % accuracy)

        batches = get_batch()
        x_dev, y_dev = pickle.load(open(test_path, "rb"))

        print("{0:-^40}".format("模型训练"))
        for data in batches:
            x_train, y_train = zip(*data)
            step, loss, accuracy = train_step(x_train, y_train)
            current_step = tf.train.global_step(sess, global_step)
            result = sess.run(merged,
                              feed_dict={lstm.input_x: x_train, lstm.input_y: y_train, lstm.dropout_keep_prob: 0.5})
            writer.add_summary(result, current_step)

            if current_step % display_train == 0:
                print("global step:{}".format(step), "==>", "loss:%.5f" % loss, "accuracy:%.5f" % accuracy)

            if current_step % display_test == 0:
                dev_step(x_dev, y_dev)
                print("")
        dev_step(x_dev, y_dev)

        saver.save(sess, ckpt_path + "/model.ckpt", global_step=current_step)

        shutil.rmtree(pb_path, ignore_errors=True)
        builder = tf.saved_model.builder.SavedModelBuilder(pb_path)
        inputs = {'input_x': tf.saved_model.utils.build_tensor_info(lstm.input_x),
                  'dropout_keep_prob': tf.saved_model.utils.build_tensor_info(lstm.dropout_keep_prob)}
        outputs = {'output': tf.saved_model.utils.build_tensor_info(lstm.probs)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], {'my_signature': signature})
        builder.save()


def main(argv=None):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    train()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run()
