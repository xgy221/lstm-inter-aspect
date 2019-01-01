import numpy as np
import tensorflow as tf

middle = []
words_list = []
words_vectors_1 = []
middle_float = []

f = open("data/glove.6B.50d.txt", "r")
lines = f.readlines()
for line in lines:
    middle = line.split()
    words_list.append(middle[0])
    middle_float = list(map(float, middle[1:]))
    words_vectors_1.append(middle_float)

words_vectors = np.array(words_vectors_1)
print(len(words_list))
print(words_list)
print(words_vectors.shape)

baseballIndex = words_list.index('the')
print(baseballIndex)
print(words_vectors[baseballIndex])

maxSeqLength = 10
numDimensions = 300
firstSentence = np.zeros(maxSeqLength, dtype='int32')
firstSentence[0] = words_list.index('i')
firstSentence[1] = words_list.index('thought')
firstSentence[2] = words_list.index('the')
firstSentence[3] = words_list.index('movie')
firstSentence[4] = words_list.index('was')
firstSentence[5] = words_list.index('incredible')
firstSentence[6] = words_list.index('and')
firstSentence[7] = words_list.index('inspiring')
print(firstSentence.shape)
print(firstSentence)

with tf.Session() as sess:
    print(tf.nn.embedding_lookup(words_vectors,firstSentence).eval().shape)

