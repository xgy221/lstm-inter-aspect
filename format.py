import numpy as np
from xml.dom.minidom import parse
import xml.dom.minidom
import os
import re
import matplotlib.pyplot as plt


def deal_glove():
    """
    处理glove, 生成中间变量
    """
    words = []
    wordVectors = []

    with open('data/glove.6B.50d.txt') as glove:
        for line in glove:
            line = line.split()
            words.append(line[0].encode('utf-8'))
            wordVectors.append(list(map(float, line[1:])))

    # 存储结果
    words = np.array(words)
    np.save('lstmSimple/cache/words.npy', words)
    wordVectors = np.array(wordVectors, dtype=np.float32)
    np.save('lstmSimple/cache/wordVectors.npy', wordVectors)


def sentiment2num(sen):
    if sen == 'positive':
        return 1
    if sen == 'negative':
        return 2
    return 0


def restaurants_train_new():
    dom_tree = xml.dom.minidom.parse(
        os.path.split(os.path.realpath(__file__))[0] + "/SemEval2014/Restaurants_Train.xml")

    sentences = list()
    _sentences = dom_tree.getElementsByTagName('sentence')
    for sentence in _sentences:
        _aspectCategories = sentence.getElementsByTagName('aspectCategory')
        aspectCategories = {}
        for aspectCategory in _aspectCategories:
            aspectCategories[aspectCategory.getAttribute('category')] = sentiment2num(
                aspectCategory.getAttribute('polarity'))
        sentences.append({
            'id': sentence.getAttribute('id'),
            'text': sentence.getElementsByTagName('text')[0].firstChild.nodeValue,
            'aspectCategories': aspectCategories
        })
    return sentences


def get_train_sentences():
    sentences = restaurants_train_new()

    return sentences[:int(len(sentences) * 0.9)]


def get_test_sentences():
    sentences = restaurants_train_new()

    return sentences[int(len(sentences) * 0.9):]


def show_sentence_len_graph():
    """
    最终决定maxSeqLength = 30 比较合理
    :return:
    """
    sentences = restaurants_train_new()
    numWords = []
    for sentence in sentences:
        numWords.append(len(str(sentence['text']).split()))

    print('count:' + str(len(numWords)))

    plt.hist(numWords, 50)
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.show()
