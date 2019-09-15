# -*- coding: utf-8 -*-

import argparse
import logging

import jieba
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.python.saved_model import tag_constants

from util import load_yaml_config, texts_to_sequences

config = load_yaml_config("config.yml")
vocab_path = config["data"]["vocab_path"]
stopword_path = config["data"]["stopword_path"]
ckpt_path = config["model"]["ckpt_path"]
pb_path = config["model"]["pb_path"]


def args_parse():
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--pb", "-pb", type=int, default=1, choices=[0, 1],
        help="[default %(default)s] default 1 use pb file, 0 use ckpt file", metavar="<PB>")

    flags = parse.parse_args()

    return flags


def predict_ckpt():
    """从检查点导入模型"""
    with tf.Session() as sess:
        checkpoint_file = tf.train.latest_checkpoint(ckpt_path)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        graph = tf.get_default_graph()
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        probs = graph.get_tensor_by_name("softmaxLayer/probs:0")

        while True:
            text = input("请输入:")
            if text == "exit":
                exit(0)
            text_seq = texts_to_sequences(text, vocab_path, stopword_path)
            pred = sess.run(probs, feed_dict={input_x: list(text_seq), keep_prob: 1.0})
            print("predict values: {}".format(pred[0]))
            print("{}".format("正向" if pred[0][0] > 0.7 else "负向" if pred[0][0] < 0.3 else "中性"))


def predict_pb():
    """从冻结图导入模型"""
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], pb_path + "/1568521090")

        graph = tf.get_default_graph()
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        probs = graph.get_tensor_by_name("softmaxLayer/probs:0")

        while True:
            text = input("请输入:")
            if text == "exit":
                exit(0)
            text_seq = texts_to_sequences(text, vocab_path, stopword_path)
            pred = sess.run(probs, feed_dict={input_x: list(text_seq), keep_prob: 1.0})
            print("predict values: {}".format(pred[0]))
            print("{}".format("正向" if pred[0][0] > 0.7 else "负向" if pred[0][0] < 0.3 else "中性"))


def main():
    flags = args_parse()
    mode = [0, 1]
    assert flags.pb in mode, ("{} not in \\[0, 1\\]".format(flags.pb))

    if flags.pb:
        predict_pb()
    else:
        predict_ckpt()


if __name__ == "__main__":
    jieba.default_logger = logging.getLogger()
    tf.logging.set_verbosity(tf.logging.ERROR)
    main()
