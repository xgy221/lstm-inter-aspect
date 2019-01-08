import numpy as np
import tensorflow as tf


class ThreeLSTM(object):
    def __init__(self, batchSize=24, da=20, ds=50, dad=50, learning_rate=0.01,
                 numClasses=3, maxSeqLength=30, l2_lambda=0., iterations=10000):
        self.batchSize = batchSize
        self.da = da
        self.ds = ds
        self.dad = dad
        self.learning_rate = learning_rate
        self.numClasses = numClasses
        self.maxSeqLength = maxSeqLength
        self.l2_lambda = l2_lambda
        self.iterations = iterations
        self.wordVectors = np.load('cache/wordVectors.npy')
        self.aspects = ['service', 'ambience', 'anecdotes/miscellaneous', 'price', 'food']
        self.cv = locals()

        self.inter_aspect_lstm()

    def calOne(self,
               aspect):  # 后面会传入方面的名字'service', 'ambience', 'anecdotes/miscellaneous', 'price', 'food'，后面注释中以food为例
        # 创建变量 food_s，这种写法作用等同为  food_s=tf.placeholder(tf.int32, [batchSize, maxSeqLength])
        self.cv[aspect + '_s'] = tf.placeholder(tf.int32,
                                                [self.batchSize,
                                                 self.maxSeqLength])  # 创建变量 food_s，作为句子的占位符，在训练的时候，需要feed进来的
        self.cv[aspect + '_w'] = tf.placeholder(tf.int32, [self.batchSize, self.maxSeqLength])  # 创建变量 food_w，作为方面的占位符

        # 填充tf占位符 food_s.shape[batchSize, maxSeqLength]=>[batchSize, maxSeqLength, wordVectorLen]
        # 我们用的是50的glove,所以wordVectorLen = 50

        data_ai = tf.nn.embedding_lookup(self.wordVectors, self.cv[aspect + '_s'])

        # 第一层lstm，套路操作，不必深究
        lstmCell_ai = tf.contrib.rnn.BasicLSTMCell(self.da)  # 创建隐藏层
        lstmCell_ai = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_ai, output_keep_prob=0.75)
        # 放在scope里面是为了防止多次调用 tf.nn.dynamic_rnn 的时候报错，不知道这么写可不可以，但是不报错了，也没见别人这么用，蛤蛤蛤

        with tf.variable_scope(aspect):
            wh = tf.get_variable(
                name='wh',
                shape=[self.maxSeqLength, 1],
                initializer=tf.random_uniform_initializer(-0.01, 0.01),
                regularizer=tf.contrib.layers.l2_regularizer(self.l2_lambda)
            )
            wb = tf.get_variable(
                name='wb',
                shape=[self.ds, self.maxSeqLength],
                initializer=tf.random_uniform_initializer(-0.01, 0.01),
                regularizer=tf.contrib.layers.l2_regularizer(self.l2_lambda)
            )

        with tf.variable_scope("lstm_a_" + aspect):
            value_ai, _ = tf.nn.dynamic_rnn(lstmCell_ai, data_ai, dtype=tf.float32)

        # 构造ti,下面又是矩阵转置，又是一顿操作，就是为了取ti，具体维度断点看
        value_ai = tf.transpose(value_ai, [1, 0, 2])
        ti = tf.gather(value_ai, int(value_ai.get_shape()[0]) - 1)

        # ---------------------------------------------------------------- 现在有ti 和 S, 开始求si
        # 把ti和s拼到一起，作为第二次lstm的输入
        data_wj = tf.nn.embedding_lookup(self.wordVectors, self.cv[aspect + '_w'])
        data_wtj = tf.concat([tf.tile(tf.expand_dims(ti, 1), [1, 30, 1]), data_wj], 2)

        # 套路操作，得到hi
        lstmCell_wtj = tf.contrib.rnn.BasicLSTMCell(self.ds)
        lstmCell_wtj = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_wtj, output_keep_prob=0.75)
        with tf.variable_scope("lstm_s_" + aspect):
            hi, _ = tf.nn.dynamic_rnn(lstmCell_wtj, data_wtj, dtype=tf.float32)
            hi = tf.transpose(hi, [0, 2, 1])

        # 创建两个临时变量，这个变量是在训练中不断优化的，对应论文中的同名变量

        # hi.shape=[batchSize, ds, maxSeqLength]    wh.shape=[maxSeqLength, 1]     M.shape=[batchSize, ds, 1]
        M = tf.tanh(tf.einsum('ijk,kl->ijl', hi, wh))

        # M.shape=[batchSize, ds, 1]    wh.shape=[ds, maxSeqLength]     alpha.shape=[batchSize, 1, maxSeqLength]
        alpha = tf.nn.softmax(tf.einsum('ijk,kl->ijl', tf.transpose(M, [0, 2, 1]), wb))
        si = tf.matmul(hi, tf.transpose(alpha, [0, 2, 1]))  # [batchSize, ds, 1]
        si = tf.transpose(si, [0, 2, 1])  # [batchSize, 1, ds]
        return si

    def inter_aspect_lstm(self):
        tf.reset_default_graph()  # tensorflow 初始化网络
        # 把所有的si拼到一起，作为第三层lstm的输入
        sis = list()
        for a in self.aspects:
            sis.append(self.calOne(a))
        sis = tf.concat(sis, 1)

        lstmCell_ad = tf.contrib.rnn.BasicLSTMCell(self.dad)
        lstmCell_ad = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_ad, output_keep_prob=0.75)
        with tf.variable_scope("lstm_ad"):
            had, hadl = tf.nn.dynamic_rnn(lstmCell_ad, sis, dtype=tf.float32)

        # 创建自调整变量
        wad = tf.get_variable(
            name='wad',
            shape=[self.numClasses, self.dad],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_lambda)
        )
        yi = tf.nn.softmax(tf.einsum('ch,bmh->bmc', wad, had))  # yi.shape = [batchSize, 5, 3]
        self.labels = tf.placeholder(tf.float32, [self.batchSize, len(self.aspects), self.numClasses])

        correctPred = tf.equal(tf.argmax(yi, 2), tf.argmax(self.labels, 2))
        self.accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))  # 本次训练的准确性

        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yi, labels=self.labels)) + sum(
            reg_loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)
        self.yi = yi
