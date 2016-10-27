from __future__ import print_function
import numpy as np
import cPickle
import pandas as pd
import sys


from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, GlobalAveragePooling1D
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional
from sklearn.preprocessing import LabelBinarizer


def padding(train, val, test, max_len):
    return sequence.pad_sequences(train, maxlen=max_len), sequence.pad_sequences(val, maxlen=max_len), sequence.pad_sequences(test, maxlen=max_len)

def build_model(params, model_name='lstm_last'):
    np.random.seed(1337)
    model = Sequential()
    model.add(Embedding(input_dim=params['embedding_input_dim'],
        output_dim=params['embedding_output_dim'],
        #mask_zero=True,
        input_length=params['embedding_input_length'],
        weights=[params['embedding_weights']]))
    if model_name == 'lstm_last':
        # LSTM layer
        # change LSTM to GRU, the network will become a Gated Recurrent Unit Based Recurrent Neural Network
        model.add(LSTM(output_dim=params['lstm_output_dim'],
        dropout_W=0.2,
        dropout_U=0.2))
    elif model_name == 'bilstm':
        model.add(Bidirectional(LSTM(output_dim=params['lstm_output_dim'],
        dropout_W=0.2,
        dropout_U=0.2)))
    elif model_name == 'lstm_average':
        model.add(LSTM(output_dim=params['lstm_output_dim'],
        dropout_W=0.2,
        dropout_U=0.2,
        return_sequences=True))
        model.add(GlobalAveragePooling1D())
    elif model_name == 'bilstm_average':
        model.add(Bidirectional(LSTM(output_dim=params['lstm_output_dim'],
        dropout_W=0.2,
        dropout_U=0.2,
        return_sequences=True)))
        model.add(GlobalAveragePooling1D())
    else:
        raise Exception("Unknown model name: %s" % model_name)
    return model


def main(model_name, dataset_name):
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
    val_set_x = []
    val_set_y = []
    test_set_x = []
    test_set_y = []

    # collect training data and testing data
    for r in rev:
        words = r['text'].split(' ')
        ids = [int(word_idx_map[w]) for w in words]
        label = int(r['y'])
        if r['type'] == 'train':
            train_set_x.append(ids)
            train_set_y.append(label)
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

    print("padding the word sequences...")
    train_set_x, val_set_x, test_set_x = padding(train=train_set_x, val=val_set_x, test=test_set_x, max_len=max_length)


    print('n_train: %s' % len(train_set_x))
    print('n_test: %s' % len(test_set_x))
    print('n_val: %s' % len(val_set_x))

    # model hyper-parameters
    params = {}
    params['embedding_input_dim'] = vocab_size + 1
    params['embedding_output_dim'] = dim_embeddings
    params['embedding_input_length'] = max_length
    params['embedding_weights'] = embeddings
    # length of hidden representation in LSTM layer
    params['lstm_output_dim'] = 128

    print("build model %s..." % model_name)
    model = build_model(params=params, model_name=model_name)
    print("add the dense layer...")
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

if __name__ == '__main__':
    model, dataset = sys.argv[1:]
    main(model_name=model, dataset_name=dataset)




