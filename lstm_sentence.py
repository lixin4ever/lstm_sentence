from __future__ import print_function
import numpy as np
import cPickle
import pandas as pd
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from sklearn.preprocessing import LabelBinarizer

dataset_name = 'semeval'
print("loading dataset %s..." % dataset_name)
# path of your dataset
x = cPickle.load(open('../RecursiveCNN/data/%s.p' % dataset_name, 'rb'))

rev, W, word_idx_map, vocab = x[0], x[1], x[2], x[3]

dim_embeddings = len(W[0])

embeddings = np.zeros((len(vocab) + 1, dim_embeddings))

for w in word_idx_map:
    wid = word_idx_map[w]
    embeddings[wid, :] = W[wid]


print("dimension of embeddings is %s" % dim_embeddings)

n_labels = len(set([r['y'] for r in rev]))

lb = LabelBinarizer()

lb.fit_transform(range(n_labels))

max_length = np.max(pd.DataFrame(rev)['num_of_words'])

print("maximum length of sentence is %s" % max_length)

vocab_size = len(vocab)

print("number of distinct words", vocab_size)

train_set_x = []
train_set_y = []
train_set_z = []
val_set_x = []
val_set_y = []
val_set_z = []
test_set_x = []
test_set_y = []
test_set_z = []

# collect training data and testing data
for r in rev:
    words = r['text'].split(' ')
    ids = [int(word_idx_map[w]) for w in words]
    label = int(r['y'])
    #strength = r['strength']
    if r['type'] == 'train':
        train_set_x.append(ids)
        train_set_y.append(label)
        #train_set_z.append(strength)
    elif r['type'] == 'val':
        val_set_x.append(ids)
        val_set_y.append(label)
    elif r['type'] == 'test':
        test_set_x.append(ids)
        test_set_y.append(label)

print("convert labels into 0-1 vectors")
train_set_y = lb.transform(train_set_y)
val_set_y = lb.transform(val_set_y)
test_set_y = lb.transform(test_set_y)
#print(test_set_y[:10])

print("padding the word sequences...")
train_set_x = sequence.pad_sequences(train_set_x, maxlen=max_length)
val_set_x = sequence.pad_sequences(val_set_x, maxlen=max_length)
test_set_x = sequence.pad_sequences(test_set_x, maxlen=max_length)
n_train = train_set_x.shape[0]

print('n_train: %s' % len(train_set_x))
print('n_test: %s' % len(test_set_x))
print('n_val: %s' % len(val_set_x))

print("build model...")
model = Sequential()
print("look up the word vectors")
# word vector look up layer
model.add(Embedding(input_dim=vocab_size + 1, 
    output_dim=dim_embeddings, 
    mask_zero=True,
    input_length=max_length,
    weights=[embeddings]))
print("run the lstm or gru")
# LSTM layer
# change LSTM to GRU, the network will become a Gated Recurrent Unit Based Recurrent Neural Network
model.add(LSTM(output_dim=128, 
    dropout_W=0.2, 
    dropout_U=0.2))
print("run the dense layer")
#model.add(Dropout(0.5))
# output dimension of the network
model.add(Dense(n_labels))
# non-linear layer
model.add(Activation('softmax'))
# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("training...")
model.fit(train_set_x, train_set_y,
          batch_size=32,
          nb_epoch=20,
          validation_data=(val_set_x, val_set_y))

score, acc = model.evaluate(test_set_x, test_set_y,
                            batch_size=32)
print('Test accuracy:', acc)






