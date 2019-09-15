# -*- coding: utf-8 -*-


import os
import pickle
import random
import shutil
import time
import datetime

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


text_cnn = __import__("model")
TextCNN = text_cnn.TextCNN

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("sequence_length", 50, "length of each sentence")
tf.app.flags.DEFINE_integer("num_classes", 2, "number of classification")
tf.app.flags.DEFINE_integer("embedding_size", 128, "word embedding size")
tf.app.flags.DEFINE_string("filter_sizes", "3,4,5", "multiply channel for filter")
tf.app.flags.DEFINE_integer("num_filters", 64, "num filter of convolution")
tf.app.flags.DEFINE_integer("batch_size", 256, "train batch size")
tf.app.flags.DEFINE_integer("train_epochs", 8, "train epochs")
tf.app.flags.DEFINE_float("dropout_prob", 0.8, "dropout keep proba")
tf.app.flags.DEFINE_float("l2_reg", 0.1, "l2 reg lambda for fully connected w")


def get_batch():
    train_x, train_y = pickle.load(open("train.pkl", "rb"))
    train_data = list(zip(train_x, train_y))

    for epoch in range(FLAGS.train_epochs):
        random.shuffle(train_data)
        for batch in range(0, len(train_data), FLAGS.batch_size):
            if batch + FLAGS.batch_size < len(train_data):
                yield train_data[batch: (batch + FLAGS.batch_size)]

def train():
    tf.reset_default_graph()
    vocab = pickle.load(open("vocab.pickle", "rb"))
    vocab_size = len(vocab.vocabulary_)
    SaveFileName = "textcnn_sentiment"  
            
    # 定义TextCnn网络
    cnn = TextCNN(
            sequence_length=FLAGS.sequence_length,
            num_classes=FLAGS.num_classes,
            vocab_size=vocab_size,
            embedding_size=FLAGS.embedding_size,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg)
    # 构建网络
    cnn.build_mode()
    
    session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        
        # 准备输出模型路径
        timestamp = str(int(time.time()))  
        # 基础目录
        SaveFileName = "textcnn_sentiment_model"
        out_dir = os.path.abspath(os.path.join(os.path.curdir, SaveFileName, timestamp))
        print("Writing to {}\n".format(out_dir))
        
        # 准备输出摘要路径
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            
        # 准备检查点名称
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # 定义保存检查点的saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        
        # 训练
        def train_step(x_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                    [cnn.train_op, cnn.global_step, cnn.train_summary_op, 
                     cnn.loss, cnn.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            train_summary_writer.add_summary(summaries, step)
            
            return (time_str, step, loss, accuracy)
        
        # 测试
        def dev_step(batch, label):
            feed_dict = {
                    cnn.input_x: batch,
                    cnn.input_y: label,
                    cnn.dropout_keep_prob: 1
                }
            step, loss, accuracy = sess.run(
                    [cnn.global_step, cnn.loss, cnn.accuracy], 
                    feed_dict)
            print("{0:-^30}".format("evaluate"))
            print("step:{}".format(step), "==>", "loss:%.5f" % loss, "accuracy:%.5f\n" % accuracy)
            
        # 获得训练测试数据
        batches = get_batch()
        x_dev, y_dev = pickle.load(open("test.pkl", "rb"))
        for data in batches:
            x_train, y_train = zip(*data)
            time_str, step, loss, accuracy = train_step(x_train, y_train)
            current_step = tf.train.global_step(sess, cnn.global_step)
            
            if step % 10 == 0:
                print("step:{}".format(step), "==>", "loss:%.5f" % loss, "accuracy:%.5f" % accuracy)
            
            if step % 100 == 0:
                saver.save(sess, checkpoint_prefix, global_step=current_step)
                
            if current_step % 50 == 0:
                dev_step(x_dev, y_dev)
        print("{0:-^30}".format("model metrics"))
        dev_step(x_dev, y_dev)
        saver.save(sess, checkpoint_prefix, global_step=current_step)
        
        # 模型保存
        model_serving_dir = os.path.join(out_dir, "tfserving")
        shutil.rmtree(model_serving_dir, ignore_errors=True)
        builder = tf.saved_model.builder.SavedModelBuilder(model_serving_dir)
        inputs = {'input_x': tf.saved_model.utils.build_tensor_info(cnn.input_x), 
                  'dropout_keep_prob': tf.saved_model.utils.build_tensor_info(cnn.dropout_keep_prob)}
        # 定义输出签名
        outputs = {'output': tf.saved_model.utils.build_tensor_info(cnn.probs)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs=outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        
        builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], {'my_signature': signature})
        builder.save()


def main(argv=None):
    train()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run()
