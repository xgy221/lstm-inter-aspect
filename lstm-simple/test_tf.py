import numpy as np
import tensorflow as tf

sess = tf.Session()

# A = tf.constant(np.arange(0, 2 * 30 * 50, dtype=np.int32), shape=[2, 50, 30])
# B = tf.constant(np.arange(0, 30, dtype=np.int32), shape=[30, 1])
#
# # t = tf.reshape(A, [-1, 30])
# # tt = tf.matmul(t, B)
# # C = tf.reshape(tt, [-1, 50, 1])
#
# C = tf.einsum('ijk,kl->ijl', A, B)
#
# a = 1

# A = tf.Variable(tf.random_normal([1, 2, 3]))
# B = tf.transpose(A, [0, 2, 1])
#
# C = tf.reshape(A, [6, 1])
# D = tf.reshape(C, [6])

# A = tf.Variable(tf.random_normal([2, 30]))
#
# C = tf.expand_dims(A, 1)
#
# B = tf.concat([C, C], 1)

A = tf.constant([
    [
        [1, 2, 3],
        [1, 4, 3],
        [11, 2, 3],
        [1, 21, 3],
        [11, 2, 31],
    ],
    [
        [1, 2, 3],
        [1, 4, 3],
        [11, 2, 3],
        [11, 2, 31],
        [111, 2, 3],
    ],
])

B = tf.constant([
    [
        [1, 2, 3],
        [1, 4, 3],
        [11, 2, 3],
        [11, 2, 3],
        [11, 2, 3],
    ],
    [
        [1, 2, 3],
        [1, 4, 3],
        [11, 2, 3],
        [11, 2, 3],
        [11, 2, 3],
    ],
])



a = 1
