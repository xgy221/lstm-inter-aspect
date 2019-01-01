import numpy as np
import format
import random
import matplotlib.pyplot as plt
import tensorflow as tf

sentences = format.restaurants_train_new()

# numWords = []
# for sentence in sentences:
#     numWords.append(len(str(sentence['text']).split()))
#
# print('count:' + str(len(numWords)))
#
# plt.hist(numWords, 50)
# plt.xlabel('Sequence Length')
# plt.ylabel('Frequency')
# plt.show()

maxSeqLength = 30
numDimensions = 300

batchSize = 24
da = 20
ds = dad = 50
# da = 100
# ds = dad = 300
numClasses = 3
iterations = 10000
aspects = ['service', 'ambience', 'anecdotes/miscellaneous', 'price', 'food']

wordVectors = np.load('cache/wordVectors.npy')
print('Loaded the word vectors!')

tf.reset_default_graph()

cv = locals()


def calOne(aspect):
    cv[aspect + '_s'] = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
    cv[aspect + '_w'] = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

    data_ai = tf.nn.embedding_lookup(wordVectors, cv[aspect + '_s'])

    lstmCell_ai = tf.contrib.rnn.BasicLSTMCell(da)
    lstmCell_ai = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_ai, output_keep_prob=0.75)
    with tf.variable_scope("lstm_a_" + aspect):
        value_ai, _ = tf.nn.dynamic_rnn(lstmCell_ai, data_ai, dtype=tf.float32)

    # weight = tf.Variable(tf.truncated_normal([da, numClasses]))
    # bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value_ai = tf.transpose(value_ai, [1, 0, 2])
    ti = tf.gather(value_ai, int(value_ai.get_shape()[0]) - 1)
    # prediction = (tf.matmul(last, weight) + bias)

    # ---------------------------------------------------------------- 现在有ti 和 S, 开始求si
    data_wj = tf.nn.embedding_lookup(wordVectors, cv[aspect + '_w'])
    data_wtj = tf.concat([tf.tile(tf.expand_dims(ti, 1), [1, 30, 1]), data_wj], 2)

    # num_units = [128, 64]
    # cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n) for n in num_units]
    # stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)

    lstmCell_wtj = tf.contrib.rnn.BasicLSTMCell(ds)
    lstmCell_wtj = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_wtj, output_keep_prob=0.75)
    with tf.variable_scope("lstm_s_" + aspect):
        hi, _ = tf.nn.dynamic_rnn(lstmCell_wtj, data_wtj, dtype=tf.float32)
        hi = tf.transpose(hi, [0, 2, 1])

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


sis = []
for a in aspects:
    sis.append(calOne(a))
sis = tf.concat(sis, 1)

lstmCell_ad = tf.contrib.rnn.BasicLSTMCell(dad)
lstmCell_ad = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_ad, output_keep_prob=0.75)
with tf.variable_scope("lstm_ad"):
    had, hadl = tf.nn.dynamic_rnn(lstmCell_ad, sis, dtype=tf.float32)

wad = tf.get_variable(
    name='wad',
    shape=[numClasses, dad],
    initializer=tf.random_uniform_initializer(-0.01, 0.01),
    regularizer=tf.contrib.layers.l2_regularizer(0.)
)
yi = tf.nn.softmax(tf.einsum('ch,bmh->bmc', wad, had))  # yi.shape = [batchSize, 5, 3]

labels = tf.placeholder(tf.float32, [batchSize, len(aspects), numClasses])

correctPred = tf.equal(tf.argmax(yi, 2), tf.argmax(labels, 2))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yi, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# ----------------------------------------------------------------------------    train start
import datetime

sess = tf.Session()
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

feedDict = {}
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

for i in range(iterations):
    # Next Batch of reviews
    ss = random.sample(sentences, batchSize)

    batchLabels = np.zeros([batchSize, len(aspects), 3])
    aspect_s = np.zeros([batchSize, maxSeqLength])
    s_i = 0
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

    sess.run(optimizer, feedDict)

    # Write summary to Tensorboard
    if i % 50 == 0:
        summary = sess.run(merged, feedDict)
        writer.add_summary(summary, i)

    # Save the network every 10,00 training iterations
    if i % 1000 == 0 and i != 0:
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)
writer.close()

a = 1
