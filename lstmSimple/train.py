import numpy as np
import format
import random
import tensorflow as tf

sentences = format.restaurants_train_new()

maxSeqLength = 30  # 由format.show_sentence_len_graph证明, 代码中既用作句子的最大长度，也用作aspect的最大长度

batchSize = 24  # tensorflow 常用方式，表示多少条输入集一起处理，24是随便取的，这里含义是多少条句子一起打包训练
da = 20  # 第一层LSTM隐藏层数，论文中指定的da=50,因为论文用的word_vector是300，我这里用的是50，所以减小到20
ds = 50  # 第二层LSTM隐藏层数
dad = 50  # 第三层LSTM隐藏层数
numClasses = 3  # positive, negative, 其他
iterations = 10000  # 训练迭代次数
# Restaurants_Train.xml 中提取出的，共有这么多种aspectCategory
aspects = ['service', 'ambience', 'anecdotes/miscellaneous', 'price', 'food']

wordVectors = np.load('cache/wordVectors.npy')
print('Loaded the word vectors!')

tf.reset_default_graph()  # tensorflow 初始化网络

cv = locals()  # 因为下面的代码中要创建动态变量，可自行百度，注释见下


def calOne(aspect):  # 后面会传入方面的名字'service', 'ambience', 'anecdotes/miscellaneous', 'price', 'food'，后面注释中以food为例
    # 创建变量 food_s，这种写法作用等同为  food_s=tf.placeholder(tf.int32, [batchSize, maxSeqLength])
    cv[aspect + '_s'] = tf.placeholder(tf.int32, [batchSize, maxSeqLength])  # 创建变量 food_s，作为句子的占位符，在训练的时候，需要feed进来的
    cv[aspect + '_w'] = tf.placeholder(tf.int32, [batchSize, maxSeqLength])  # 创建变量 food_w，作为方面的占位符

    # 填充tf占位符 food_s.shape[batchSize, maxSeqLength]=>[batchSize, maxSeqLength, wordVectorLen]
    # 我们用的是50的glove,所以wordVectorLen = 50
    data_ai = tf.nn.embedding_lookup(wordVectors, cv[aspect + '_s'])

    # 第一层lstm，套路操作，不必深究
    lstmCell_ai = tf.contrib.rnn.BasicLSTMCell(da)  # 创建隐藏层
    lstmCell_ai = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_ai, output_keep_prob=0.75)
    # 放在scope里面是为了防止多次调用 tf.nn.dynamic_rnn 的时候报错，不知道这么写可不可以，但是不报错了，也没见别人这么用，蛤蛤蛤
    with tf.variable_scope("lstm_a_" + aspect):
        value_ai, _ = tf.nn.dynamic_rnn(lstmCell_ai, data_ai, dtype=tf.float32)

    # 构造ti,下面又是举证转置，又是一顿操作，就是为了取ti，具体维度断点看
    value_ai = tf.transpose(value_ai, [1, 0, 2])
    ti = tf.gather(value_ai, int(value_ai.get_shape()[0]) - 1)

    # ---------------------------------------------------------------- 现在有ti 和 S, 开始求si
    # 把ti和s拼到一起，作为第二次lstm的输入
    data_wj = tf.nn.embedding_lookup(wordVectors, cv[aspect + '_w'])
    data_wtj = tf.concat([tf.tile(tf.expand_dims(ti, 1), [1, 30, 1]), data_wj], 2)

    # 套路操作，得到hi
    lstmCell_wtj = tf.contrib.rnn.BasicLSTMCell(ds)
    lstmCell_wtj = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_wtj, output_keep_prob=0.75)
    with tf.variable_scope("lstm_s_" + aspect):
        hi, _ = tf.nn.dynamic_rnn(lstmCell_wtj, data_wtj, dtype=tf.float32)
        hi = tf.transpose(hi, [0, 2, 1])

    # 创建两个临时变量，这个变量是在训练中不断优化的，对应论文中的同名变量
    with tf.variable_scope(aspect):
        wh = tf.get_variable(
            name='wh',
            shape=[maxSeqLength, 1],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(0.)
        )
        wb = tf.get_variable(
            name='wb',
            shape=[ds, maxSeqLength],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(0.)
        )

    # hi.shape=[batchSize, ds, maxSeqLength]    wh.shape=[maxSeqLength, 1]     M.shape=[batchSize, ds, 1]
    M = tf.tanh(tf.einsum('ijk,kl->ijl', hi, wh))

    # M.shape=[batchSize, ds, 1]    wh.shape=[ds, maxSeqLength]     alpha.shape=[batchSize, 1, maxSeqLength]
    alpha = tf.nn.softmax(tf.einsum('ijk,kl->ijl', tf.transpose(M, [0, 2, 1]), wb))
    si = tf.matmul(hi, tf.transpose(alpha, [0, 2, 1]))  # [batchSize, ds, 1]
    si = tf.transpose(si, [0, 2, 1])  # [batchSize, 1, ds]

    # ---------------------------------------------------------------- 现在有si
    return si


# 把所有的si拼到一起，作为第三层lstm的输入
sis = []
for a in aspects:
    sis.append(calOne(a))
sis = tf.concat(sis, 1)

# 自行体会
lstmCell_ad = tf.contrib.rnn.BasicLSTMCell(dad)
lstmCell_ad = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_ad, output_keep_prob=0.75)
with tf.variable_scope("lstm_ad"):
    had, hadl = tf.nn.dynamic_rnn(lstmCell_ad, sis, dtype=tf.float32)

# 创建自调整变量
wad = tf.get_variable(
    name='wad',
    shape=[numClasses, dad],
    initializer=tf.random_uniform_initializer(-0.01, 0.01),
    regularizer=tf.contrib.layers.l2_regularizer(0.)
)
yi = tf.nn.softmax(tf.einsum('ch,bmh->bmc', wad, had))  # yi.shape = [batchSize, 5, 3]

# 整个网络的输出，很重要，最后训练的时候，要填充的就是所有的 tf.placeholder，前面的都是输入，这个是输出的结果
labels = tf.placeholder(tf.float32, [batchSize, len(aspects), numClasses])

# 这俩玩意是方便可视化展示的，训练中可以不要
correctPred = tf.equal(tf.argmax(yi, 2), tf.argmax(labels, 2))  # 计算得到预测正确的结果矩阵
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))  # 得到本次训练的准确性

# 套路操作
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yi, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# ---    train start, 构造输入和输出，feed到网络里，训练
import datetime

# 建立tf会话，只有在会话中，tf才会具体计算值，包括各种占位符和变量
sess = tf.Session()

# 构造可视化图标变量
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
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
aspect_w = np.zeros([batchSize, maxSeqLength])
a_ids = np.zeros([maxSeqLength])
for a in aspects:
    a_w = a.split()
    w_i = 0
    for w in a_w:
        try:
            a_ids[w_i] = wordsList.index(w)
        except ValueError:
            a_ids[w_i] = 399999  # Vector for unkown words
        w_i += 1
for i in range(batchSize):
    aspect_w[i] = a_ids
for a in aspects:
    feedDict[cv[a + '_w']] = aspect_w

# 开始执行指定次数的迭代
for i in range(iterations):
    # ----------------------------------------------------构造句子输入和label输入

    # 随机从句子中取 batchSize 个作为本次训练的输入句子
    ss = random.sample(sentences, batchSize)

    batchLabels = np.zeros([batchSize, len(aspects), 3])
    aspect_s = np.zeros([batchSize, maxSeqLength])
    s_i = 0
    # 遍历句子，得到此次训练的句子输入和label输入
    for s in ss:
        s_w = s['text'].split()
        s_ids = np.zeros([maxSeqLength])
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
        for a in aspects:
            sentiment = s['aspectCategories'][a] if s['aspectCategories'].__contains__(a) else 0
            batchLabels[s_i][a_i][sentiment] = 1
            feedDict[cv[a + '_s']] = aspect_s
            a_i += 1
        s_i += 1
    feedDict[labels] = batchLabels

    # ruuuuuuuuun
    sess.run(optimizer, feedDict)

    # Write summary to Tensorboard，每50个做一个输出指标，用于Tensorboard展示
    if i % 50 == 0:
        summary = sess.run(merged, feedDict)
        writer.add_summary(summary, i)

    # Save the network every 1000 training iterations, 保存训练中间结果
    if i % 1000 == 0 and i != 0:
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)
writer.close()
