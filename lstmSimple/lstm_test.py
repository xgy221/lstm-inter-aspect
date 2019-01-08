import lstmSimple.lstm_network as xgy
import datetime
import tensorflow as tf
import numpy as np
import random
import format

sentences = format.restaurants_train_new()
ln = xgy.ThreeLSTM()

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

wordsList = np.load('cache/words.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()  # Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8

# 很重要，这个就是所有placeholder要填充到的地方
feedDict = {}

# ---------------------------------------------------因为每次输入的时候，无论怎么在sentences中取batchSize，方面输入都是固定的，所以先初始化好，
aspect_w = np.zeros([ln.batchSize, ln.maxSeqLength])
a_ids = np.zeros([ln.maxSeqLength])
for a in ln.aspects:
    a_w = a.split()
    w_i = 0
    for w in a_w:
        try:
            a_ids[w_i] = wordsList.index(w)
        except ValueError:
            a_ids[w_i] = 399999  # Vector for unkown words
        w_i += 1
for i in range(ln.batchSize):
    aspect_w[i] = a_ids
for a in ln.aspects:
    feedDict[ln.cv[a + '_w']] = aspect_w

# 开始执行指定次数的迭代
for i in range(10):
    # ----------------------------------------------------构造句子输入和label输入

    # 随机从句子中取 batchSize 个作为本次训练的输入句子
    ss = random.sample(sentences, ln.batchSize)

    batchLabels = np.zeros([ln.batchSize, len(ln.aspects), 3])
    aspect_s = np.zeros([ln.batchSize, ln.maxSeqLength])
    s_i = 0
    # 遍历句子，得到此次训练的句子输入和label输入
    for s in ss:
        s_w = s['text'].split()
        s_ids = np.zeros([ln.maxSeqLength])
        w_i = 0
        for w in s_w:
            try:
                s_ids[w_i] = wordsList.index(w)
            except ValueError:
                s_ids[w_i] = 399999  # Vector for unkown words
            w_i += 1
            if w_i >= 30:
                break
        aspect_s[s_i] = s_ids

        bl = []
        a_i = 0
        for a in ln.aspects:
            sentiment = s['aspectCategories'][a] if s['aspectCategories'].__contains__(a) else 0
            batchLabels[s_i][a_i][sentiment] = 1
            feedDict[ln.cv[a + '_s']] = aspect_s
            a_i += 1
        s_i += 1
    feedDict[ln.labels] = batchLabels

    print("Accuracy for this batch:", (sess.run(ln.accuracy, feedDict)) * 100)
