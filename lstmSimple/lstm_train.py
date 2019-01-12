import lstmSimple.lstm_network as xgy
import datetime
import tensorflow as tf
import numpy as np
import random
import format

sentences = format.get_train_sentences()
ln = xgy.ThreeLSTM()

# ---    train start, 构造输入和输出，feed到网络里，训练

# 建立tf会话，只有在会话中，tf才会具体计算值，包括各种占位符和变量
sess = tf.Session()

# 构造可视化图标变量
tf.summary.scalar('Loss', ln.loss)
tf.summary.scalar('Accuracy', ln.accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

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
for i in range(ln.iterations):
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

    # ruuuuuuuuun
    sess.run(ln.optimizer, feedDict)

    # Write summary to Tensorboard，每50个做一个输出指标，用于Tensorboard展示
    if i % 50 == 0:
        summary = sess.run(merged, feedDict)
        writer.add_summary(summary, i)

    # Save the network every 1000 training iterations, 保存训练中间结果
    if i % 1000 == 0 and i != 0:
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)
writer.close()
