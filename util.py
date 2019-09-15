# -*- coding: utf-8 -*-


import jieba
import yaml
from tensorflow.contrib import learn


def load_yaml_config(yaml_file):
    with open(yaml_file) as f:
        config = yaml.load(f)
    return config


def texts_to_sequences(text, vocab_path, stopword_path):
    vocab = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    stopword = [line.strip() for line in open(stopword_path, 'r', encoding='utf-8').readlines()]

    # 输入是一个可迭代对象，输出是generator[array]
    text_seq = vocab.transform(
        (" ".join([x for x in jieba.cut(text) if x not in stopword]),))

    return text_seq
