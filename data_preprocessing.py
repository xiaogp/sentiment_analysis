# -*- coding: utf-8 -*-


import pickle
import logging
import os

import jieba
import tensorflow as tf
from sklearn.model_selection import train_test_split

from util import load_yaml_config

config = load_yaml_config("config.yml")
seq_length = config["model"]["seq_length"]
min_frequency = config["data"]["min_frequency"]
stopword_path = config["data"]["stopword_path"]
positive_path = config["data"]["positive_path"]
negative_path = config["data"]["negative_path"]
vocab_path = config["data"]["vocab_path"]
train_path = config["data"]["train_path"]
test_path = config["data"]["test_path"]


def load_stopwords():
    stopword_list = [line.replace("\n", "") for line in open(
        stopword_path, "r", encoding="utf8").readlines()]
    return stopword_list


def load_data():
    with open(positive_path, "r", encoding="utf8") as f1, open(negative_path, "r", encoding="utf8") as f2:
        positive = [line.strip() for line in f1.readlines()]
        negative = [line.strip() for line in f2.readlines()]
        positive_label = [[1, 0]] * len(positive)
        negative_label = [[0, 1]] * len(negative)

        corpus = positive + negative
        labels = positive_label + negative_label
        print("{0:-^30}".format("样本预览"))
        print("正样本:", len(positive))
        print("负样本:", len(negative))

        return corpus, labels


def seg(corpus):
    stopwords_list = load_stopwords()
    seg_list = [" ".join([w for w in jieba.cut(words) if w not in stopwords_list]) for words in corpus]
    return seg_list


def tokenizer():
    corpus, labels = load_data()
    seg_list = seg(corpus)
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(seq_length, min_frequency)
    onehot = vocab_processor.fit_transform(seg_list)
    vocab_processor.save(vocab_path)
    print("{0:-^60}".format("number of words : %d" % len(vocab_processor.vocabulary_)))

    for i, key in enumerate(vocab_processor.vocabulary_._mapping):
        print(key, vocab_processor.vocabulary_._mapping[key])
        if i == 5:
            break

    train_x, test_x, train_y, test_y = train_test_split(list(onehot), labels, test_size=0.1, random_state=1)

    print("{0:-^60}".format("one sample overview"))
    print(train_x[0], train_y[0])

    pickle.dump((train_x, train_y), open(train_path, "wb"))
    pickle.dump((test_x, test_y), open(test_path, "wb"))


def main(argv=None):
    tokenizer()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)
    jieba.setLogLevel(logging.INFO)
    tf.app.run()
