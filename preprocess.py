import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
from gensim.models import Word2Vec
import math

def build_dataset(text_path, label_path, dataset, clear_string=True):
    """
    built the dataset
    :param text_path:
    :param label_path:
    :param dataset:
    :return:
    """
    revs = []
    labels = []
    with open(label_path, 'r') as fp:
        for line in fp:
            labels.append(line.strip())
    ratings = []
    strengths = []
    if rating_path:
        with open(rating_path, 'r') as fp:
            for line in fp:
                values = line.strip().split(' ')[1:]
                ratings.append([int(v) for v in values])
    if strength_path:
        with open(strength_path, 'r') as fp:
            for line in fp:
                values = line.strip().split(' ')
                strengths.append([float(v) for v in values])
    count = 0
    vocab = defaultdict(int)

    val_count = 0
    # default division of dataset: train/val/test = 6/2/2
    n_samples = len(labels)
    train_count = n_samples * 0.6
    val_count = n_samples * 0.2
    tr = 0
    te = 0
    val = 0
    max_length = 0
    max_sentence_count = 0
    total_token_count = 0.0
    with open(text_path, 'r') as fp:
        for line in fp:
            text = line.strip()
            if clear_string and dataset != 'sina':
                text = clean_str(text)
            words = text.split(' ')
            if len(words) > max_sentence_count:
                max_sentence_count = len(words)
            total_token_count += len(words)
            label = labels[count]

            #rating = "unused"
            for word in set(words):
                vocab[word] += 1
            length = len(words)
            content = text
            if count < train_count:
                if strength:
                    datanum = {'y':label, 'text':content, 'num_of_words':length, 'type':'train', 'strength': strength}
                else:
                    datanum = {'y':label, 'text':content, 'num_of_words':length, 'type':'train'}
                tr += 1
            elif val_count:
                if count < train_count + val_count:
                    if strength:
                        datanum = {'y':label, 'text':content, 'num_of_words':length, 'type':'val', 'strength': strength}
                    else:
                        datanum = {'y':label, 'text':content, 'num_of_words':length, 'type':'val'}
                    val += 1
                else:
                    if strength:
                        datanum = {'y':label, 'text':content, 'num_of_words':length, 'type':'test', 'strength': strength}
                    else:
                        datanum = {'y':label, 'text':content, 'num_of_words':length, 'type':'test'}
                    te += 1
            else:
                te += 1
                if strengths:
                    datanum = {'y':label, 'text':content, 'num_of_words':length, 'type':'test', 'strength': strength}
                else:
                    datanum = {'y':label, 'text':content, 'num_of_words':length, 'type':'test'}
            revs.append(datanum)
            count += 1
    assert len(revs) == len(labels)
    print "max length of sentence is", max_sentence_count
    print "average length of sentence is", total_token_count / float(len(revs))
    return revs, vocab, bt_vocab

def get_word_vector(word_vecs, k=300):
    """
    build indexed word vectors and mapping between word and word id
    :param word_vecs:
    :param k:
    :return:
    """
    vocab_size = len(word_vecs)
    word_to_id = dict()
    indexed_word_vectors = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    indexed_word_vectors[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        indexed_word_vectors[i] = word_vecs[word]
        #if i == 1:
        #    print word_vecs[word]
        word_to_id[word] = i
        i += 1
    return indexed_word_vectors, word_to_id

def load_pretrained_word_vector(fname, vocab, dataset):
    """
    load pre-trained word vectors
    :param fname: name of word2vec file
    :param vocab:
    :param languange:
    :return:
    """
    word_vecs = {}
    if topic_modeling:
        word_to_vectors = cPickle.load(open(fname, 'rb'))
        for word in vocab:
            if word in word_to_vectors:
                word_vecs[word] = word_to_vectors[word]
    else:
        if dataset == 'sina' and False:
            model = Word2Vec.load('D:/LIXIN/LIXIN_DATA/Chinese_word2vec/wiki.zh.text.model')
            for word in vocab:
                if word in model:
                    word_vecs[word] = model[word]
        else:
            with open(fname, "rb") as f:
                header = f.readline()
                vocab_size, layer1_size = map(int, header.split())
                binary_len = np.dtype('float32').itemsize * layer1_size
                for line in xrange(vocab_size):
                    word = []
                    while True:
                        # read one character from file
                        ch = f.read(1)
                        if ch == ' ':
                            word = ''.join(word)
                            break
                        if ch != '\n':
                            word.append(ch)
                    if word in vocab and vocab[word] >= 2:
                        word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                    else:
                        f.read(binary_len)
    return word_vecs

def add_unseen_embedding(word_vecs, vocab, min_df=1, k=300):
    """
    randomly initialized unseen word vectors
    :param word_vecs:
    :param vocab:
    :param min_df:
    :param k:
    :return:
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            # randomize initialization
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    #if not ("'s" in string or "'ve" in string or "'t" in string or "'re" in string or "'d" in string or "'ll" in string):

    special_map = {"'s":"dot_s", "'ve":"dot_ve", "n't":"dot_n't", "'re":"dot_re", "'d":"dot_re", "'ll":"dot_ll"}
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " %s" % special_map["'s"], string)
    string = re.sub(r"\'ve", " %s" % special_map["'ve"], string)
    string = re.sub(r"n\'t", " %s" % special_map["n't"], string)
    string = re.sub(r"\'re", " %s" % special_map["'re"], string)
    string = re.sub(r"\'d", " %s" % special_map["'d"], string)
    string = re.sub(r"\'ll", " %s" % special_map["'ll"], string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"`", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.replace("'", "")

    for (k, v) in special_map.iteritems():
        string = string.replace(v, k)

    # convert uppercase letter into downcase letter
    #print string.strip().lower()
    return string.strip() if TREC else string.strip().lower()

if __name__ == '__main__':
    word2vec_file = 'GoogleNews-vectors-negative300.bin'
    # flag that denotes cross domain testing
    dataset = 'MySpace'
    text_path = './%s/%s_text.txt' % (dataset, dataset)
    label_path = './%s/%s_label.txt' % (dataset, dataset)
    rating_path = './%s/%s_rating.txt' % (dataset, dataset)
    strength_path = './%s/%s_strength.txt' % (dataset, dataset)
    print 'load dataset from %s, %s and %s' % (text_path, label_path, rating_path)
    revs, vocab = build_dataset(text_path=text_path, label_path=label_path, dataset=dataset, clear_string=True)
    max_l = np.max(pd.DataFrame(revs)['num_of_words'])
    print "data loaded"
    print 'number of sentences is', len(revs)
    print 'vocab size is:', len(vocab)
    print 'load word to vector from', word2vec_file
    word_vecs = load_pretrained_word_vector(fname=word2vec_file, vocab=vocab, dataset=dataset)

    print 'word vectors loaded'
    print 'number of seen words is:', len(word_vecs)

    add_unseen_embedding(word_vecs=word_vecs, vocab=vocab, k=300, min_df=1)
    indexed_word_vecs, word_to_id = get_word_vector(word_vecs, k=300)
    rand_vecs = {}
    add_unseen_embedding(word_vecs=rand_vecs, vocab=vocab, k=300, min_df=1)
    indexed_rand_vecs, _ = get_word_vector(rand_vecs, k=300)
    cPickle.dump([revs, indexed_word_vecs, word_to_id, vocab], open('./%s/%s.p' % (dataset, dataset), 'wb'))
    #print indexed_word_vecs[1]
    print 'dataset created!'