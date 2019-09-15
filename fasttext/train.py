# -*- coding: utf-8 -*-


import os
import re
import time
import subprocess

import jieba
import fasttext
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


def preprocess():
    stopword = []
    with open("stopwords.txt") as f:
        for line in f:
            stopword.append(line.strip())

    with open("fasttext_new.txt", "a") as f1:
        with open("comments_positive.txt", "r") as f2:
            for line in f2:
                cut = " ".join([x for x in jieba.cut(line) if x not in stopword])
                new = "__label__1 " + cut.strip()
                f1.write(new + "\n")

    with open("fasttext_new.txt", "a") as f1:
        with open("comments_negative.txt", "r") as f2:
            for line in f2:
                cut = " ".join([x for x in jieba.cut(line) if x not in stopword])
                new = "__label__0 " + cut.strip()
                f1.write(new + "\n")

    # 对原文件进行shuffle核数据分割
    _, line = subprocess.getstatusoutput("cat fasttext_new.txt | wc -l")
    train_line = int(int(line) * 0.9)
    subprocess.call("shuf fasttext_new.txt >> fasttext_shuf.txt", shell=True)
    subprocess.call("split -l {} fasttext_shuf.txt split".format(train_line), shell=True)
    subprocess.call("mv splitaa train.txt", shell=True)
    subprocess.call("mv splitab test.txt", shell=True)
    subprocess.call("rm -rf splita*", shell=True)


def build_model():
    start = time.time()
    model = fasttext.train_supervised('train.txt')
    print("{0:-^30}".format("模型训练"))
    print("elapse time: %.3fs" % (time.time() - start))
    model.save_model("fasttext_model.bin")


def model_metrics(model):
    test = []
    label = []
    with open("test.txt", "r") as f:
        for line in f:
            string = re.match("__label__\d(.*)", line).group(1).strip()
            lab = re.match("__label__\d", line).group()
            test.append(string)
            label.append(lab)

    def label_transform(x):
        if "1" in x:
            return 1
        return 0

    predict = list(map(lambda x: x[0], model.predict(test)[0]))
    print("{0:-^30}".format("模型测试"))
    print("accuracy: %.3f" % (accuracy_score(label, predict)))
    print("precision: %.3f" % (precision_score(list(map(label_transform, label)), list(map(label_transform, predict)))))
    print("recall: %.3f" % (recall_score(list(map(label_transform, label)), list(map(label_transform, predict)))))

    print("{0:-^30}".format("混淆矩阵"))
    print(confusion_matrix(label, predict))

    print("{0:-^30}".format("预览预测结果"))
    for i, j in enumerate(predict):
        print("文本: {}\t".format(test[i]))
        print("实际: {}\t".format(label[i]))
        print("预测: {}\n".format(j))
        if i == 2:
            break


def load_model():
    model = fasttext.load_model("fasttext_model.bin")
    model_metrics(model)


if __name__ == "__main__":
    if not os.path.exists("fasttext_new.txt"):
        preprocess()
    if not os.path.exists("fasttext_model.bin"):
        build_model()
    load_model()
