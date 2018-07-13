# -*- coding: utf-8 -*-
"""
 @version: python2.7
 @author: luofuli
 @time: 2017/7/1
"""

import numpy as np
from utils import path
_path = path.WSD_path()
data_dir = _path.DATA_DIR


def load_glove(path=None, dim=None, size=None):   # download from https://nlp.stanford.edu/projects/glove/
    if path is None:
        if size=='6B':
            path = data_dir + 'glove.6B/glove.6B.' + str(dim) + 'd.txt'
        elif size=='42B' and dim==300:
            path = data_dir+'glove.42B.300d.txt'
        elif size=='840B' and dim==300:
            path = data_dir+'glove.840B.300d.txt'
        else:
            print(u'No pre-trained word-embeddings at dir:%s' % (data_dir))
            exit(-3)

    wordvecs = {}
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            tokens = line.split(' ')
            vec = np.array(tokens[1:], dtype=np.float32)
            wordvecs[tokens[0]] = vec

    return wordvecs


def fill_with_gloves(word_to_id, path=None, emb_size=None, vocab_size=None, wordvecs=None):
    if not wordvecs:
        wordvecs = load_glove(path, emb_size, vocab_size)

    n_words = len(word_to_id)

    if emb_size is None:
        emb_size = len(wordvecs[wordvecs.keys()[0]])

    res = np.zeros([n_words, emb_size], dtype=np.float32)
    n_not_found = 0
    words_notin = set()
    for word, id in word_to_id.iteritems():
        if '#' in word:
            word = word.split('#')[0]   # Remove pos tag

        if '-' in word:
            words = word.split('-')
        elif '_' in word:
            words = word.split('_')
        else:
            words = [word]

        vecs = []
        for w in words:
            if w in wordvecs:
                vecs.append(wordvecs[w])    # add word2vec for multi-word
        if vecs != []:
            res[id, :] = np.mean(np.array(vecs), 0)
        else:
            words_notin.add(word)
            n_not_found += 1
            res[id, :] = np.random.normal(0.0, 0.1, emb_size)
    print 'n words not found in glove word vectors: ' + str(n_not_found)
    open('../tmp/word_not_in_glove.txt','w').write((u'\n'.join(words_notin)).encode('utf-8'))

    return res


