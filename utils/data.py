# -*- coding: utf-8 -*-
"""
 @version: python2.7
 @author: luofuli
 @time: 2017/7/1
"""

import lxml.etree as et
import math
import numpy as np
import collections
import re
import random
from bs4 import BeautifulSoup
from bs4 import NavigableString
import pickle
from utils import path
from nltk.corpus.reader.wordnet import WordNetCorpusReader
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()  # download wordnet: import nltk; nltk.download("wordnet") in readme.txt

_path = path.WSD_path()
wn = WordNetCorpusReader(_path.WORDNET_PATH, '.*')
print('wordnet version %s: %s' % (wn.get_version(), _path.WORDNET_PATH))

path_words_notin_vocab = '../tmp/words_notin_vocab_{}.txt'

pos_dic = {
    'ADJ': u'a',
    'ADV': u'r',
    'NOUN': u'n',
    'VERB': u'v', }

POS_LIST = pos_dic.values()  # ['a', 'r', 'n', 'v']


def load_train_data(dataset):
    if dataset in _path.LS_DATASET:
        return load_lexical_sample_data(_path.LS_TRAIN_PATH.format(dataset), True)
    elif dataset in _path.ALL_WORDS_TRAIN_DATASET:
        return load_all_words_data(_path.ALL_WORDS_TRAIN_PATH.format(dataset),
                                   _path.ALL_WORDS_TRAIN_KEY_PATH.format(dataset),
                                   _path.ALL_WORDS_DIC_PATH.format(dataset), True)
    else:
        raise ValueError('%s, %s. Provided: %s' % (
            ','.join(_path.LS_DATASET), ','.join(_path.ALL_WORDS_TRAIN_DATASET), dataset))


def load_val_data(dataset):
    if dataset in _path.LS_DATASET:
        return load_lexical_sample_data(_path.LS_VAL_PATH.format(dataset), True)
    elif dataset in _path.ALL_WORDS_TEST_DATASET:
        return load_all_words_data(_path.ALL_WORDS_TEST_PATH.format(dataset),
                                   _path.ALL_WORDS_TEST_KEY_PATH.format(dataset), None, False)
    else:
        raise ValueError('%s, %s. Provided: %s' % (
            ','.join(_path.LS_DATASET), ','.join(_path.ALL_WORDS_TEST_DATASET), dataset))


def load_test_data(dataset):
    if dataset in _path.LS_DATASET:
        return load_lexical_sample_data(_path.LS_TEST_PATH.format(dataset), False)
    elif dataset in _path.ALL_WORDS_TEST_DATASET:
        return load_all_words_data(_path.ALL_WORDS_TEST_PATH.format(dataset),
                                   _path.ALL_WORDS_TEST_KEY_PATH.format(dataset), None, False)
    else:
        raise ValueError('%s, %s. Provided: %s' % (
            ','.join(_path.LS_DATASET), ','.join(_path.ALL_WORDS_TEST_DATASET), dataset))


def load_lexical_sample_data(path, is_training=None):
    data = []
    doc = BeautifulSoup(open(path), 'xml')
    instances = doc.find_all('instance')

    for instance in instances:
        answer = None
        context = None
        for child in instance.children:
            if isinstance(child, NavigableString):
                continue
            if child.name == 'answer':
                senseid = child.get('senseid')
                if senseid == 'P' or senseid == 'U':
                    pass
                elif not answer:
                    answer = senseid
            elif child.name == 'context':
                context = child.prettify()
            else:
                print(child.name)
                print(instance.text)
                raise ValueError('unknown child tag to instance')

        def clean_context(ctx_in, has_target=False):
            replace_target = re.compile("<head.*?>.*</head>")
            replace_newline = re.compile("\n")
            replace_dot = re.compile("\.")
            replace_cite = re.compile("'")
            replace_frac = re.compile("[\d]*frac[\d]+")
            replace_num = re.compile("\s\d+\s")
            rm_context_tag = re.compile('<.{0,1}context>')
            rm_cit_tag = re.compile('\[[eb]quo\]')
            rm_misc = re.compile("[\[\]\$`()%/,\.:;-]")

            ctx = replace_newline.sub(' ', ctx_in)  # (' <eop> ', ctx)
            if not has_target:
                ctx = replace_target.sub(' <target> ', ctx)

            ctx = replace_dot.sub(' ', ctx)  # .sub(' <eos> ', ctx)
            ctx = replace_cite.sub(' ', ctx)  # .sub(' <cite> ', ctx)
            ctx = replace_frac.sub(' <frac> ', ctx)
            ctx = replace_num.sub(' <number> ', ctx)
            ctx = rm_cit_tag.sub(' ', ctx)
            ctx = rm_context_tag.sub('', ctx)
            ctx = rm_misc.sub('', ctx)

            word_list = [word for word in re.split('`|, | +|\? |! |: |; |\(|\)|_|,|\.|"|“|”|\'|\'', ctx.lower()) if word]
            return word_list

        # if valid
        if (is_training and answer and context) or (not is_training and context):
            context = clean_context(context)
            lemma = instance.get('id').split('.')[0]
            pos = instance.get('id').split('.')[1]
            if pos in POS_LIST:
                word = lemma + '#' + pos
            else:
                word = lemma
            pos_list = ['<pad>'] * len(context)
            x = {
                'id': instance.get('id'),
                'context': context,
                'target_sense': answer,  # don't support multiple answers
                'target_word': word,
                'poss': pos_list,
            }

            data.append(x)

    return data


def load_all_words_data(data_path, key_path=None, dic_path=None, is_training=False):
    word_count_info = {}
    if dic_path:
        soup = BeautifulSoup(open(dic_path), 'lxml')
        for lexelt_tag in soup.find_all('lexelt'):
            lemma = lexelt_tag['item']
            sense_count_wn = int(lexelt_tag['sence_count_wn'])
            sense_count_corpus = int(lexelt_tag['sense_count_corpus'])
            word_count_info[lemma] = [sense_count_wn, sense_count_corpus]

    id_to_sensekey = {}
    if key_path:
        for line in open(key_path).readlines():
            id = line.split()[0]
            sensekey = line.split()[1]  # multiple sense
            id_to_sensekey[id] = sensekey

    context = et.iterparse(data_path, tag='sentence')

    data = []
    poss = set()
    for event, elem in context:
        sent_list = []
        pos_list = []
        for child in elem:
            word = child.get('lemma').lower()
            sent_list.append(word)
            pos = child.get('pos')
            pos_list.append(pos)
            poss.add(pos)

        i = -1
        for child in elem:
            if child.tag == 'wf':
                i += 1
            elif child.tag == 'instance':
                i += 1
                id = child.get('id')
                lemma = child.get('lemma').lower()
                if '(' in lemma:
                    print id
                pos = child.get('pos')
                word = lemma + '#' + pos_dic[pos]
                if key_path:
                    sensekey = id_to_sensekey[id]
                else:
                    sensekey = None
                if is_training:
                    if word_count_info[word][0] <= 1 or word_count_info[word][1] <= 1:
                        continue

                context = sent_list[:]
                if context[i] != lemma:
                    print '/'.join(context)
                    print i
                    print lemma
                context[i] = '<target>'

                x = {
                    'id': id,
                    'context': context,
                    'target_sense': sensekey,  # don't support multiple answers
                    'target_word': word,
                    'poss': pos_list,
                }

                data.append(x)

    if is_training:
        poss_list = ['<pad>', '<eos>', '<unk>'] + list(sorted(poss))
        # print 'Wirting to tmp/pos_dic.pkl:' + ' '.join(poss_list)
        poss_map = dict(zip(poss_list, range(len(poss_list))))
        with open('../tmp/pos_dic.pkl', 'wb') as f:
            pickle.dump((poss_map), f)

    return data


def filter_word_and_sense(train_data, test_data, min_sense_freq=1, max_n_sense=40):
    train_words = set()
    for elem in train_data:
        train_words.add(elem['target_word'])

    test_words = set()
    for elem in test_data:
        test_words.add(elem['target_word'])

    target_words = train_words & test_words

    counter = collections.Counter()
    for elem in train_data:
        if elem['target_word'] in target_words:
            counter.update([elem['target_sense']])

    # remove infrequent sense
    filtered_sense = [item for item in counter.items() if item[1] >= min_sense_freq]

    count_pairs = sorted(filtered_sense, key=lambda x: -x[1])
    senses, _ = list(zip(*count_pairs))
    all_sense_to_id = dict(zip(senses, range(len(senses))))

    word_to_senses = {}
    for elem in train_data:
        target_word = elem['target_word']
        target_sense = elem['target_sense']

        if target_sense in all_sense_to_id:
            if target_word not in word_to_senses:
                word_to_senses.update({target_word: [target_sense]})
            else:
                if target_sense not in word_to_senses[target_word]:
                    word_to_senses[target_word].append(target_sense)

    filtered_word_to_sense = {}
    for target_word, senses in word_to_senses.iteritems():
        senses = sorted(senses, key=lambda s: all_sense_to_id[s])
        senses = senses[:max_n_sense]
        if len(senses) > 1:  # must leave more than one sense
            np.random.shuffle(senses)  # shuffle senses to avoid MFS
            filtered_word_to_sense[target_word] = senses

    return filtered_word_to_sense


def data_postprocessing(train_dataset, test_dataset, train_data, test_data, back_off_type="FS",
                        min_sense_freq=1, max_n_sense=40):
    filtered_word_to_sense = filter_word_and_sense(train_data, test_data, min_sense_freq, max_n_sense)

    new_train_data = []
    for elem in train_data:
        target_word = elem['target_word']
        target_sense = elem['target_sense']
        if target_word in filtered_word_to_sense and target_sense in filtered_word_to_sense[target_word]:
            new_train_data.append(elem)

    new_test_data = []
    for elem in test_data:
        target_word = elem['target_word']
        target_sense = elem['target_sense']
        if target_word in filtered_word_to_sense and target_sense in filtered_word_to_sense[target_word]:
        # test will ignore sense not in train
            new_test_data.append(elem)

    mfs_key_path = _path.MFS_PATH.format(train_dataset)
    fs_key_path = _path.WNFS_PATH

    mfs_id_key_map = {}
    for line in open(mfs_key_path):
        id = line.split()[0]
        key = line.split()[1]
        mfs_id_key_map[id] = key
    fs_id_key_map = {}
    for line in open(fs_key_path):
        id = line.split()[0]
        key = line.split()[1]
        fs_id_key_map[id] = key

    back_off_result = []

    mfs_using_fs_info = 0
    target_word_back_off = set()
    all_target_words = set()
    for elem in test_data:
        target_word = elem['target_word']
        target_sense = elem['target_sense']
        all_target_words.add(target_word)
        if target_word not in filtered_word_to_sense:
            target_word_back_off.add(target_word)
            if test_dataset != _path.ALL_WORDS_TEST_DATASET[0]:  # ALL dataset id format has dataset name
                id = test_dataset + '.' + elem['id']
            else:
                id = elem['id']
            if back_off_type == 'FS':
                back_off_result.append([elem['id'], fs_id_key_map[id]])
            if back_off_type == 'MFS':  # dataset MFS may not cover all-words
                if id in mfs_id_key_map:
                    back_off_result.append([elem['id'], mfs_id_key_map[id]])
                else:
                    mfs_using_fs_info += 1
                    back_off_result.append([elem['id'], fs_id_key_map[id]])

    print('***MFS Using wordnet information instance number:%d ' % (mfs_using_fs_info))
    print('***Using back off target words: %s/%s' % (len(target_word_back_off), len(all_target_words)))

    back_off_result_path = _path.BACK_OFF_RESULT_PATH.format(back_off_type)
    print('***Writing to back_off_results to file:%s' % back_off_result_path)
    with open(back_off_result_path, 'w') as f:
        for instance_id, predicted_sense in back_off_result:
            f.write('%s %s\n' % (instance_id, predicted_sense))

    return new_train_data, new_test_data, filtered_word_to_sense, back_off_result


def data_postprocessing_for_validation(val_data, filtered_word_to_sense=None):
    new_val_data = []
    for elem in val_data:
        target_word = elem['target_word']
        target_sense = elem['target_sense']
        if target_word in filtered_word_to_sense and target_sense in filtered_word_to_sense[target_word]:
            new_val_data.append(elem)
    return new_val_data


def test_data_postprocessing(train_dataset, train_target_words, test_data, back_off_type='MFS'):
    key_path = None
    if back_off_type == 'MFS':
        key_path = _path.MFS_PATH.format(train_dataset)
    elif back_off_type == 'FS':
        key_path = _path.WNFS_PATH

    id_key_map = {}
    if key_path:
        for line in open(key_path):
            id = line.split()[0]
            key = line.split()[1]
            id_key_map[id] = key

    back_off_result = []
    new_test_data = []
    for d in test_data:
        if d['target_word'] in train_target_words:
            new_test_data.append(d)
        else:
            id = d['id']
            if id in id_key_map:
                back_off_result.append([id, id_key_map[id]])
    return new_test_data, back_off_result


def build_vocab(data):
    """
    :param data: list of dicts containing attribute 'context'
    :return: a dict with words as key and ids as value
    """
    counter = collections.Counter()
    for elem in data:
        counter.update(elem['context'])
        counter.update([elem['target_word']])

    # remove infrequent words
    min_freq = 1
    filtered = [item for item in counter.items() if item[1] >= min_freq]

    count_pairs = sorted(filtered, key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    add_words = ['<pad>', '<eos>', '<unk>'] + list(words)
    word_to_id = dict(zip(add_words, range(len(add_words))))

    return word_to_id


def build_sense_ids(word_to_senses):

    words = list(word_to_senses.keys())
    target_word_to_id = dict(zip(words, range(len(words))))

    target_sense_to_id = [dict(zip(word_to_senses[word], range(len(word_to_senses[word])))) for word in words]

    n_senses_from_word_id = dict([(target_word_to_id[word], len(word_to_senses[word])) for word in words])
    return target_word_to_id, target_sense_to_id, n_senses_from_word_id, word_to_senses


class Instance:
    pass


def convert_to_numeric(data, word_to_id, target_word_to_id, target_sense_to_id,
                       ignore_sense_not_in_train=True, mode=''):
    words_notin_vocab = []

    with open('../tmp/pos_dic.pkl', 'rb') as f:
        pos_to_id = pickle.load(f)

    all_data = []
    target_tag_id = word_to_id['<target>']
    instance_sensekey_not_in_train = []
    for insi, instance in enumerate(data):
        words = instance['context']
        poss = instance['poss']
        assert len(poss) == len(words)
        ctx_ints = []
        pos_ints = []

        for i, word in enumerate(words):
            if word in word_to_id:
                ctx_ints.append(word_to_id[word])
                pos_ints.append(pos_to_id[poss[i]])
            elif len(word) > 0:
                ctx_ints.append(word_to_id['<unk>'])
                pos_ints.append(pos_to_id['<unk>'])
                words_notin_vocab.append(word)

        stop_idx = ctx_ints.index(target_tag_id)
        xf = np.array(ctx_ints[:stop_idx], dtype=np.int32)
        pf = np.array(pos_ints[:stop_idx], dtype=np.int32)
        xb = np.array(ctx_ints[stop_idx + 1:], dtype=np.int32)[::-1]
        pb = np.array(pos_ints[stop_idx + 1:], dtype=np.int32)[::-1]

        instance_id = instance['id']
        target_word = instance['target_word']
        target_sense = instance['target_sense']

        try:
            target_id = target_word_to_id[target_word]
            senses = target_sense_to_id[target_id]
        except KeyError as e:
            # print e
            continue

        if target_sense in senses:  # test will ignore sense not in train, same as data_postprocessing
            sense_id = senses[target_sense]
        else:
            instance_sensekey_not_in_train.append([instance_id, target_sense])
            if ignore_sense_not_in_train:
                continue
            else:
                sense_id = 0  # sensekey not in train is classified as label 0

        instance = Instance()
        instance.id = instance_id
        instance.xf = xf
        instance.xb = xb
        instance.pf = pf
        instance.pb = pb
        instance.target_word_id = word_to_id[target_word]
        instance.target_pos_id = pos_ints[stop_idx]
        instance.target_id = target_id
        instance.sense_id = sense_id

        all_data.append(instance)

    tmp_lenth = len(instance_sensekey_not_in_train)
    if tmp_lenth:
        print('###%s instance_sensekey_not_in_train: %s' % (mode, tmp_lenth))

    store_notin_vocab_words(words_notin_vocab, mode=mode)
    print('%s words_notin_vocab:%d' % (mode, len(words_notin_vocab)))

    return all_data


def store_notin_vocab_words(words_notin_vocab, mode='', clean=True):
    if clean:
        old = []
    else:
        try:
            old = open(path_words_notin_vocab.format(mode)).read()
            old = old.split('\n')
        except Exception as e:
            old = []

    ws = []
    for word in words_notin_vocab:
        try:
            word = word.decode('utf-8')
            ws.append(word)
        except Exception as e:
            continue
    new = set(ws + old)
    open(path_words_notin_vocab.format(mode), 'w').write('\n'.join(new))


def batch_generator(is_training, batch_size, data, dict_data, pad_id, n_step_f, n_step_b, pad_last_batch=False):
    data_len = len(data)
    n_batches_float = data_len / float(batch_size)
    n_batches = int(math.ceil(n_batches_float)) if pad_last_batch else int(n_batches_float)

    if is_training:
        random.shuffle(data)

    for i in range(n_batches):
        batch = data[i * batch_size:(i + 1) * batch_size]

        # context word
        xfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        xbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        xfbs = np.zeros([batch_size, n_step_f + n_step_b + 1], dtype=np.int32)
        xfs.fill(pad_id)
        xbs.fill(pad_id)
        xfbs.fill(pad_id)

        # context pos
        pfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        pbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        pfbs = np.zeros([batch_size, n_step_f + n_step_b + 1], dtype=np.int32)  # 0 is pad for pos, no need pad_id

        # x forward backward
        for j in range(batch_size):
            if i * batch_size + j < data_len:
                n_to_use_f = min(n_step_f, len(batch[j].xf))
                n_to_use_b = min(n_step_b, len(batch[j].xb))
                if n_to_use_f:
                    xfs[j, -n_to_use_f:] = batch[j].xf[-n_to_use_f:]
                    pfs[j, -n_to_use_f:] = batch[j].pf[-n_to_use_f:]
                if n_to_use_b:
                    xbs[j, -n_to_use_b:] = batch[j].xb[-n_to_use_b:]
                    pbs[j, -n_to_use_b:] = batch[j].pb[-n_to_use_b:]
                xfbs[j] = np.concatenate((xfs[j], [batch[j].target_word_id], xbs[j][::-1]), axis=0)
                pfbs[j] = np.concatenate((pfs[j], [batch[j].target_pos_id], pbs[j][::-1]), axis=0)

        # id
        instance_ids = [inst.id for inst in batch]

        # labels
        target_ids = [inst.target_id for inst in batch]
        sense_ids = [inst.sense_id for inst in batch]

        if len(target_ids) < batch_size:  # padding
            n_pad = batch_size - len(target_ids)
            # print('Batch padding size: %d'%(n_pad))
            target_ids += [0] * n_pad
            sense_ids += [0] * n_pad
            instance_ids += [0] * n_pad  # instance_ids += [''] * n_pad

        target_ids = np.array(target_ids, dtype=np.int32)
        sense_ids = np.array(sense_ids, dtype=np.int32)

        glosses_ids = [dict_data[0][target_ids[i]] for i in range(batch_size)]
        glosses_lenth = [dict_data[1][target_ids[i]] for i in range(batch_size)]
        sense_mask = [dict_data[2][target_ids[i]] for i in range(batch_size)]

        yield (xfs, xbs, xfbs, pfs, pbs, pfbs, target_ids,
               sense_ids, instance_ids, glosses_ids, glosses_lenth, sense_mask)


def batch_generator_hyp(is_training, batch_size, data, dict_data, pad_id, n_step_f, n_step_b, n_hyper, n_hypo,
                        pad_last_batch=False):
    data_len = len(data)
    n_batches_float = data_len / float(batch_size)
    n_batches = int(math.ceil(n_batches_float)) if pad_last_batch else int(n_batches_float)

    if is_training:
        random.shuffle(data)

    for i in range(n_batches):
        batch = data[i * batch_size:(i + 1) * batch_size]

        # context word
        xfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        xbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        xfbs = np.zeros([batch_size, n_step_f + n_step_b + 1], dtype=np.int32)
        xfs.fill(pad_id)
        xbs.fill(pad_id)
        xfbs.fill(pad_id)

        # context pos
        pfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        pbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        pfbs = np.zeros([batch_size, n_step_f + n_step_b + 1], dtype=np.int32)  # 0 is pad for pos, no need pad_id

        # x forward backward
        for j in range(batch_size):
            if i * batch_size + j < data_len:
                n_to_use_f = min(n_step_f, len(batch[j].xf))
                n_to_use_b = min(n_step_b, len(batch[j].xb))
                if n_to_use_f:
                    xfs[j, -n_to_use_f:] = batch[j].xf[-n_to_use_f:]
                    pfs[j, -n_to_use_f:] = batch[j].pf[-n_to_use_f:]
                if n_to_use_b:
                    xbs[j, -n_to_use_b:] = batch[j].xb[-n_to_use_b:]
                    pbs[j, -n_to_use_b:] = batch[j].pb[-n_to_use_b:]
                xfbs[j] = np.concatenate((xfs[j], [batch[j].target_word_id], xbs[j][::-1]), axis=0)
                pfbs[j] = np.concatenate((pfs[j], [batch[j].target_pos_id], pbs[j][::-1]), axis=0)

        # id
        instance_ids = [inst.id for inst in batch]

        # labels
        target_ids = [inst.target_id for inst in batch]
        sense_ids = [inst.sense_id for inst in batch]

        if len(target_ids) < batch_size:  # padding
            n_pad = batch_size - len(target_ids)
            # print('Batch padding size: %d'%(n_pad))
            target_ids += [0] * n_pad
            sense_ids += [0] * n_pad
            instance_ids += [0] * n_pad  # instance_ids += [''] * n_pad

        target_ids = np.array(target_ids, dtype=np.int32)
        sense_ids = np.array(sense_ids, dtype=np.int32)

        # [gloss_to_id, gloss_lenth, sense_mask, hyper_lenth, hypo_lenth]
        glosses_ids = [dict_data[0][target_ids[i]] for i in
                       range(batch_size)]  # [batch_size, max_n_sense, n_hyp, max_gloss_words]
        glosses_lenth = [dict_data[1][target_ids[i]] for i in range(batch_size)]  # [batch_size, max_n_sense]
        sense_mask = [dict_data[2][target_ids[i]] for i in range(batch_size)]  # [batch_size, max_n_sense, mask_size]
        hyper_lenth = [dict_data[3][target_ids[i]] for i in range(batch_size)]  # [batch_size, max_n_sense]
        hypo_lenth = [dict_data[4][target_ids[i]] for i in range(batch_size)]  # [batch_size, max_n_sense]

        yield (xfs, xbs, xfbs, pfs, pbs, pfbs, target_ids, sense_ids, instance_ids, glosses_ids,
               glosses_lenth, sense_mask, hyper_lenth, hypo_lenth)


# get gloss from dictionary.xml
def load_dictionary(dataset, target_words=None, expand_type=0, n_hyper=3, n_hypo=3):
    gloss_dic = {}
    if dataset in _path.LS_DATASET:
        dic_path = _path.LS_DIC_PATH.format(dataset)
        target_words = None  # Lexical task don't need target words filter
    elif dataset in _path.ALL_WORDS_TRAIN_DATASET:
        dic_path = _path.ALL_WORDS_DIC_PATH.format(dataset)
    else:
        raise ValueError(
            '%s or %s. Provided: %s' % (','.join(_path.LS_DATASET), ','.join(_path.ALL_WORDS_TRAIN_DATASET), dataset))

    soup = BeautifulSoup(open(dic_path), 'lxml')
    all_sense_tag = soup.find_all('sense')
    for sense_tag in all_sense_tag:
        id = sense_tag['id']
        key = id  # all-words
        # key = id.replace("-", "'")  # senseval2_LS README EG:pull_in_one-s_horns%2:32:00::
        gloss = sense_tag['gloss']
        if expand_type in [1, 2, 3]:
            gloss = expand_gloss(key, expand_type, n_hyper, n_hypo)
        elif expand_type == 4:
            gloss = expand_gloss_list(key, n_hyper, n_hypo)
        if target_words:  # for all_words task
            target_word = sense_tag.parent['item']
            if target_word in target_words:
                gloss_dic[id] = gloss
        else:  # for lexical example task
            gloss_dic[id] = gloss
    return gloss_dic


def expand_gloss(key, expand_type, n_hyper, n_hypo):
    try:
        lemma = wn.lemma_from_key(key)
    except Exception as e:
        print e
        print key
        exit(-1)
    synset = lemma.synset()
    if expand_type == 1:  # 'hyper':
        h = list(synset.closure(lambda s: s.hypernyms(), n_hyper))[:n_hyper]
    elif expand_type == 2:  # 'hypo
        h = list(synset.closure(lambda s: s.hypernyms(), n_hyper))[:n_hyper]
    else:  # 'hyper' + 'hypo
        h1 = list(synset.closure(lambda s: s.hypernyms(), n_hypo))[:n_hypo]
        h2 = list(synset.closure(lambda s: s.hypernyms(), n_hypo))[:n_hypo]
        h2.reverse()
        h = h1 + h2

    glosses = [synset.definition()]
    for i, s in enumerate(h):
        glosses.append(s.definition())
    if expand_type == 3 and h != []:
        glosses.append(synset.definition())  # target->hyper->hypo->target
    r = ' <eos> '.join(glosses)
    return r


def expand_gloss_list(key, n_hyper, n_hypo):
    lemma = wn.lemma_from_key(key)
    synset = lemma.synset()
    hyper = list(synset.closure(lambda s: s.hypernyms(), n_hyper))[:n_hyper]
    hypo = list(synset.closure(lambda s: s.hyponyms(), n_hypo))[:n_hypo]
    # hyper.reverse()   # No need

    glosses = [''] * (n_hyper + n_hypo + 1)  # hyper3 hyper2 hyper1 target hypo1 hypo2 hypo3
    for i, s in enumerate(hyper):
        glosses[n_hyper - 1 - i] = s.definition()  # reverse is here
    glosses[n_hyper] = synset.definition()
    for i, s in enumerate(hypo):
        glosses[n_hyper + 1 + i] = s.definition()  # no reverse

    return glosses  # length: [n_hypo+1+n_hyper]


def split_sentence(sent):
    sent = re.findall(r"[\w]+|[^\s\w]", sent)
    for i, word in enumerate(sent):
        sent[i] = wordnet_lemmatizer.lemmatize(word)
    return sent


# make initial sense id(in dataset) to new sense id, and make numeric for gloss defination
def bulid_dictionary_id(gloss_dict, target_sense_to_id, word_to_id, pad_id, mask_size, max_gloss_words=100):
    # t_max_gloss_words = max([len(split_sentence(g)) for g in gloss_dict.values()])
    # print('original max_gloss_words: %s' % (t_max_gloss_words))
    n_target_words = len(target_sense_to_id)
    print('n_target_words: %s' % n_target_words)
    max_n_sense = max([len(sense_to_ids) for sense_to_ids in target_sense_to_id])
    print('max_n_sense %d' % (max_n_sense))
    gloss_to_id = np.zeros([n_target_words, max_n_sense, max_gloss_words], dtype=np.int32)
    gloss_to_id.fill(pad_id)

    words_notin_vocab = []

    gloss_lenth = np.zeros([n_target_words, max_n_sense], dtype=np.int32)
    sense_mask = np.zeros([n_target_words, max_n_sense, mask_size], dtype=np.int32)
    for i, sense_to_ids in enumerate(target_sense_to_id):
        if i % 500 == 0:
            print("Bulid dictionary: %s/%s" % (i, len(target_sense_to_id)))
        for id0 in sense_to_ids:  # id0 is the initial id in dataset
            j = sense_to_ids[id0]
            gloss_words = split_sentence(gloss_dict[id0])
            sense_mask[i][j][:] = 1
            words = []
            for word in gloss_words:
                if word in word_to_id:
                    words.append(word_to_id[word])
                elif len(word) > 0:
                    words.append(word_to_id['<unk>'])
                    words_notin_vocab.append(word)

            words = words[:max_gloss_words]

            if len(words) > 0:
                gloss_to_id[i, j, :len(words)] = words  # pad in the end
                gloss_lenth[i][j] = len(words)
    store_notin_vocab_words(words_notin_vocab, mode='gloss')
    print('%s words_notin_vocab:%d' % ('gloss', len(words_notin_vocab)))
    return [gloss_to_id, gloss_lenth, sense_mask], max_n_sense


def bulid_dictionary_id_hyp(gloss_dict, target_sense_to_id, word_to_id, pad_id, mask_size, max_gloss_words, n_hyper, n_hypo):
    n_target_words = len(target_sense_to_id)
    n_hy = n_hyper + n_hypo + 1
    print('n_target_words: %s' % n_target_words)
    max_n_sense = max([len(sense_to_ids) for sense_to_ids in target_sense_to_id])
    print('max_n_sense %d' % (max_n_sense))
    gloss_to_id = np.zeros([n_target_words, max_n_sense, n_hy, max_gloss_words], dtype=np.int32)
    gloss_to_id.fill(pad_id)

    words_notin_vocab = []

    gloss_lenth = np.zeros([n_target_words, max_n_sense, n_hy], dtype=np.int32)
    hyper_lenth = np.zeros([n_target_words, max_n_sense], dtype=np.int32)
    hypo_lenth = np.zeros([n_target_words, max_n_sense], dtype=np.int32)
    sense_mask = np.zeros([n_target_words, max_n_sense, mask_size], dtype=np.int32)
    for i, sense_to_ids in enumerate(target_sense_to_id):
        if i % 500 == 0:
            print("Bulid dictionary: %s/%s" % (i, len(target_sense_to_id)))
        for senseid in sense_to_ids:  # senseid is like 'have%2:40:00::'
            j = sense_to_ids[senseid]
            gloss_list = gloss_dict[senseid]
            sense_mask[i][j][:] = 1
            for k, gloss in enumerate(gloss_list):
                if gloss == ['']:
                    continue

                gloss_words = split_sentence(gloss)
                words = []
                for word in gloss_words:
                    if word in word_to_id:
                        words.append(word_to_id[word])
                    elif len(word) > 0:
                        words.append(word_to_id['<unk>'])
                        words_notin_vocab.append(word)

                words = words[:max_gloss_words]

                if len(words) > 0:
                    gloss_to_id[i, j, k, :len(words)] = words  # pad in the end
                    gloss_lenth[i][j][k] = len(words)
                    hyper_lenth[i][j] = max(hyper_lenth[i][j], n_hyper - k + 1)  # +1 is gloss of the target word
                    hypo_lenth[i][j] = max(hypo_lenth[i][j], k - n_hyper + 1)

    store_notin_vocab_words(words_notin_vocab, mode='gloss_hyp')
    print('%s words_notin_vocab:%d' % ('hyp gloss', len(words_notin_vocab)))
    return [gloss_to_id, gloss_lenth, sense_mask, hyper_lenth, hypo_lenth], max_n_sense
