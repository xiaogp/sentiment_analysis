# -*- coding: utf-8 -*-

import os
import logging
import time
import argparse
import multiprocessing

import jieba
from gensim.models import word2vec

from util import load_yaml_config

config = load_yaml_config("config.yml")
positive_path = config["data"]["positive_path"]
negative_path = config["data"]["negative_path"]
stopword_path = config["data"]["stopword_path"]
seg_path = config["data"]["seg_path"]


def args_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--embedding_size", "-es", type=int, default=128,
        help="[default %(default)s] presto host")
    parse.add_argument(
        "--min_count", "-mc", type=int, default=3,
        help="[default %(default)s] min word frequency for model")
    parse.add_argument(
        "--sg", "-sg", type=int, default=1,
        help="[default %(default)s] CBOW:0, skip-gram:1")
    parse.add_argument(
        "--iter", "-it", type=int, default=5,
        help="[default %(default)s] train iter")

    flags = parse.parse_args()

    return flags


def preprocess():
    """将中文txt文件提前分词好去除停用词，用" "隔开"""
    stopword = [line.strip() for line in open(stopword_path, 'r', encoding='utf-8').readlines()]

    text_list = [positive_path, negative_path]
    for text in text_list:
        with open(seg_path, "a") as f1:
            with open(text, "r") as f2:
                for line in f2:
                    cut = " ".join([x for x in jieba.cut(line.replace("\n", "")) if x not in stopword])
                    f1.write(cut + "\n")


def train():
    """
    直接训练txt文本文件，保存模型二进制文件，和词向量文本文件
    word2vec args:
        size: 词向量维度
        window：窗口长度，默认是5
        min_count：最小词频，默认是5
        workers：使用多少线程
        sg：0是CBOW，1是skip—gram，默认是0，CBOW更快，skip-gram更准
        iter：训练epochs，默认是5，大预料可以适当增大
    """
    flags = args_parse()
    print("{0:-^60}".format("参数解析"))
    print(flags)
    sentences = word2vec.LineSentence(seg_path)  # 直接读取text的行
    start = time.time()
    model = word2vec.Word2Vec(sentences, size=flags.embedding_size, window=5,
                              min_count=flags.min_count, sg=flags.sg,
                              workers=multiprocessing.cpu_count())
    print("elapse time: %.3fs" % (time.time() - start))
    print("word counts: %d" % len(model.wv.vocab.keys()))
    model.save("embedding.model")
    model.wv.save_word2vec_format("embedding.txt", binary=False)  # 将每个词和他的向量保存在txt文件


def demo():
    """
    通过训练好的模型，实现
    (1)计算单个词最相似的词topn
    (2)计算两个词的相似度
    (3)对应关系推演
    """
    model = word2vec.Word2Vec.load("embedding.model")
    while True:
        query = input("请输入内容：")
        if query == "exit":
            exit(0)

        query_list = query.split()
        assert len(query_list) <= 3, "您输入的词太多，请重新输入"

        try:
            if len(query_list) == 1:
                print("{0:-^30}".format("最相似的10个词"))
                for word, score in model.most_similar(query_list[0]):
                    print(word, score)

            elif len(query_list) == 2:
                print("{0:-^30}".format("两个词的相似度"))
                cos_similary = model.similarity(query_list[0], query_list[1])
                print(cos_similary)

            else:
                print("{0:-^30}".format("关系推理"))
                res = model.most_similar([query_list[0], query_list[1]], [query_list[2]], topn=10)
                for word, score in res:
                    print("{} => {} ≈ {} => {}: {}".format(query_list[0], query_list[1], query_list[2], word, score))
        except Exception as e:
            print("您输入的词不在词表")


if __name__ == "__main__":
    jieba.setLogLevel(logging.INFO)
    if not os.path.exists("embedding.txt"):
        preprocess()
    if not os.path.exists("embedding.model"):
        train()
    demo()
