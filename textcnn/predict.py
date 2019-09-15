# -*- coding: utf-8 -*-


import timeit
import logging

import jieba
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.python.saved_model import tag_constants

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("text", "tmp", "new comment text")


def text_clean():
    vocab = learn.preprocessing.VocabularyProcessor.restore("vocab.pickle")
    
    stopword = set()
    with open("stopwords.txt", 'r', encoding='utf-8') as f:
        for line in f:
            stopword.add(line.strip())
    new_comment = vocab.transform((" ".join([x for x in jieba.cut(FLAGS.text) if x not in stopword]), ))
    
    return new_comment


def predict_from_saver():
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], 'textcnn_sentiment_model/1568547209/tfserving')
        graph = tf.get_default_graph()  

        input_x = graph.get_operation_by_name("input_x").outputs[0]
        keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        probs = graph.get_tensor_by_name("output/probs:0")
    
        start = timeit.default_timer()
        new_comment = text_clean()
        pred = sess.run(probs, feed_dict={input_x: list(new_comment), keep_prob: 1.0})
        print("predict values: {}".format(pred[0]))
        print("{}".format("正向" if pred[0][0] > 0.7 else "负向" if pred[0][0] < 0.3 else "中性"))
            

def main(argv=None):
    predict_from_saver()


if __name__ == "__main__":
    jieba.default_logger = logging.getLogger()
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run()
