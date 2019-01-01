import numpy as np
from xml.dom.minidom import parse
import xml.dom.minidom
import os
import re


def deal_glove():
    """
    处理glove
    """
    words = []
    wordVectors = []

    with open('data/glove.6B.50d.txt') as glove:
        for line in glove:
            line = line.split()
            words.append(line[0].encode('utf-8'))
            wordVectors.append(list(map(float, line[1:])))

    # 存储结果
    # words = np.array(words)
    # np.save('lstm-simple/cache/words.npy', words)
    # wordVectors = np.array(wordVectors, dtype=np.float32)
    # np.save('lstm-simple/cache/wordVectors.npy', wordVectors)


def deal_absa_2015_restaurants_trial():
    """
    处理restaurants
    """
    # dom_tree = xml.dom.minidom.parse("SemEval2015/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml")
    dom_tree = xml.dom.minidom.parse("SemEval2015/absa-2015_restaurants_trial.xml")

    sentences = list()
    _sentences = dom_tree.getElementsByTagName('sentence')
    for sentence in _sentences:
        _opinions = sentence.getElementsByTagName('Opinion')
        opinions = list()
        for opinion in _opinions:
            opinions.append({
                'target': opinion.getAttribute('target'),
                'category': opinion.getAttribute('category'),
                'polarity': opinion.getAttribute('polarity'),
                'from': opinion.getAttribute('from'),
                'to': opinion.getAttribute('to'),
            })
        sentences.append({
            'id': sentence.getAttribute('id'),
            'text': sentence.getElementsByTagName('text')[0].firstChild.nodeValue,
            'opinions': opinions
        })
    return sentences


def restaurants_train():
    # dom_tree = xml.dom.minidom.parse("SemEval2015/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml")
    dom_tree = xml.dom.minidom.parse(
        os.path.split(os.path.realpath(__file__))[0] + "/SemEval2014/Restaurants_Train.xml")

    sentences = list()
    _sentences = dom_tree.getElementsByTagName('sentence')
    for sentence in _sentences:
        _aspectTerms = sentence.getElementsByTagName('aspectTerm')
        aspectTerms = list()
        for aspectTerm in _aspectTerms:
            aspectTerms.append({
                'term': aspectTerm.getAttribute('term'),
                'polarity': aspectTerm.getAttribute('polarity'),
                'from': aspectTerm.getAttribute('from'),
                'to': aspectTerm.getAttribute('to'),
            })
        sentences.append({
            'id': sentence.getAttribute('id'),
            'text': sentence.getElementsByTagName('text')[0].firstChild.nodeValue,
            'aspectTerms': aspectTerms
        })
    return sentences


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


def cleanSentences(string):
    # Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


# res = deal_absa_2015_restaurants_trial()
# res = restaurants_train()

# import tensorflow as tf
#
# normal = tf.truncated_normal([25, 2])
#
a = 1
