import nltk
import json
from math import log10


def generate_confusion_matrix(error_path,
                              ins_confusion_path='./confusion_matrix/ins_confusion_matrix.txt',
                              del_confusion_path='./confusion_matrix/del_confusion_matrix.txt',
                              sub_confusion_path='./confusion_matrix/sub_confusion_matrix.txt',
                              trans_confusion_path='./confusion_matrix/trans_confusion_matrix.txt'):
    error_f = open(error_path, 'r')

    ins_dict = {}  # store the matrix with dictionary, for example, {"ge": 1} means matrix[g][e] = 1
    del_dict = {}
    sub_dict = {}
    trans_dict = {}

    lines = error_f.readlines()
    for line in lines:
        parts = line.split(':')
        correct_word = parts[0].replace('\n', '').replace(' ', '')
        wrong_words = [word.replace('\n', '').replace(' ', '') for word in parts[1].split(',')]
        for wrong_word in wrong_words:
            ed_matrix = get_min_ed_matrix(wrong_word, correct_word)
            new_ins_dict, new_del_dict, new_sub_dict, new_trans_dict = backtrace(ed_matrix, wrong_word, correct_word)
            for key, value in new_ins_dict.items():
                ins_dict[key] = ins_dict.get(key, 0) + value
            for key, value in new_del_dict.items():
                del_dict[key] = del_dict.get(key, 0) + value
            for key, value in new_sub_dict.items():
                sub_dict[key] = sub_dict.get(key, 0) + value
            for key, value in new_trans_dict.items():
                trans_dict[key] = trans_dict.get(key, 0) + value

    # Save as json format
    json.dump(ins_dict, open(ins_confusion_path, 'w'))
    json.dump(del_dict, open(del_confusion_path, 'w'))
    json.dump(sub_dict, open(sub_confusion_path, 'w'))
    json.dump(trans_dict, open(trans_confusion_path, 'w'))

    error_f.close()


def generate_confusion_logp(corpus_str,
                            ins_count_confusion_path='./confusion_matrix/addconfusion.data',
                            del_count_confusion_path='./confusion_matrix/delconfusion.data',
                            sub_count_confusion_path='./confusion_matrix/subconfusion.data',
                            trans_count_confusion_path='./confusion_matrix/revconfusion.data',
                            ins_confusion_path='./confusion_matrix/ins_confusion_matrix.txt',
                            del_confusion_path='./confusion_matrix/del_confusion_matrix.txt',
                            sub_confusion_path='./confusion_matrix/sub_confusion_matrix.txt',
                            trans_confusion_path='./confusion_matrix/trans_confusion_matrix.txt'):
    ins_count_dict = json.load(open(ins_count_confusion_path))
    del_count_dict = json.load(open(del_count_confusion_path))
    sub_count_dict = json.load(open(sub_count_confusion_path))
    trans_count_dict = json.load(open(trans_count_confusion_path))
    ins_logp_dict = {}
    del_logp_dict = {}
    sub_logp_dict = {}
    trans_logp_dict = {}

    # Get the characters set
    ch_set = set()
    for key in ins_count_dict.keys():
        ch_set |= set(key)
    for key in del_count_dict.keys():
        ch_set |= set(key)
    for key in sub_count_dict.keys():
        ch_set |= set(key)
    for key in trans_count_dict.keys():
        ch_set |= set(key)

    # Preprocess the corpus
    sents_str = ''
    for sent in nltk.tokenize.sent_tokenize(corpus_str):
        words = nltk.tokenize.word_tokenize(sent)
        sents_str = sents_str + ' ' + ' '.join(words)
    total_ch_count = len(sents_str)

    # Make the Confusion Matrix
    for ch1 in ch_set:
        for ch2 in ch_set:
            # Ins
            ins_count = ins_count_dict.get(ch1 + ch2, 0) + 1    # add 1 smooth
            corpus_count = sents_str.count(ch1) if ch1 != '#' else sents_str.count(' ')
            ins_logp_dict[ch1 + ch2] = log10(ins_count / corpus_count) if corpus_count >= ins_count else log10(1 / total_ch_count)
            # Del
            del_count = del_count_dict.get(ch1 + ch2, 0) + 1
            corpus_count = corpus_count = sents_str.count(ch1 + ch2) if ch1 != '#' else sents_str.count(' ' + ch2)
            del_logp_dict[ch1 + ch2] = log10(del_count / corpus_count) if corpus_count >= del_count else log10(1 / total_ch_count)
            # Sub
            sub_count = sub_count_dict.get(ch1 + ch2, 0) + 1
            corpus_count = sents_str.count(ch2)
            sub_logp_dict[ch1 + ch2] = log10(sub_count / corpus_count) if corpus_count >= sub_count else log10(1 / total_ch_count)
            # Trans
            trans_count = trans_count_dict.get(ch1 + ch2, 0) + 1
            corpus_count = sents_str.count(ch1 + ch2)
            trans_logp_dict[ch1 + ch2] = log10(trans_count / corpus_count) if corpus_count >= trans_count else log10(1 / total_ch_count)
    ins_logp_dict['<unk>'] = log10(1 / total_ch_count)
    del_logp_dict['<unk>'] = log10(1 / total_ch_count)
    sub_logp_dict['<unk>'] = log10(1 / total_ch_count)
    trans_logp_dict['<unk>'] = log10(1 / total_ch_count)

    # Save as json format
    json.dump(ins_logp_dict, open(ins_confusion_path, 'w'))
    json.dump(del_logp_dict, open(del_confusion_path, 'w'))
    json.dump(sub_logp_dict, open(sub_confusion_path, 'w'))
    json.dump(trans_logp_dict, open(trans_confusion_path, 'w'))


def get_min_ed_matrix(s1, s2):
    s1_len = len(s1)
    s2_len = len(s2)

    # Initialize Matrix
    ed_matrix = [[0 for j in range(s2_len + 1)] for i in range(s1_len + 1)]

    for i in range(s1_len + 1):
        ed_matrix[i][0] = i
    for j in range(s2_len + 1):
        ed_matrix[0][j] = j

    # Dynamic Program
    for i in range(1, s1_len + 1):
        for j in range(1, s2_len + 1):
            before = ed_matrix[i][j]
            comp = []
            # ins
            comp.append(ed_matrix[i][j - 1] + 1)
            # del
            comp.append(ed_matrix[i - 1][j] + 1)
            # sub
            if s1[i - 1] == s2[j - 1]:
                comp.append(ed_matrix[i - 1][j - 1])
            else:
                comp.append(ed_matrix[i - 1][j - 1] + 1)
            # trans
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                comp.append(ed_matrix[i - 2][j - 2] + 1)
            ed_matrix[i][j] = min(comp)
            after = ed_matrix[i][j]

    return ed_matrix


def backtrace(ed_matrix, s1, s2):
    s1_len = len(s1)
    s2_len = len(s2)

    ins_dict = {}
    del_dict = {}
    sub_dict = {}
    trans_dict = {}

    i = s1_len
    j = s2_len
    while i != 0 or j != 0:
        # trans?
        if i > 1 and j > 1 and ed_matrix[i][j] == ed_matrix[i - 2][j - 2] + 1:
            trans_dict[f"{s2[j - 2:j]}"] = trans_dict.get(f"{s2[j - 2:j]}", 0) + 1
            i -= 2
            j -= 2
            continue
        # sub?
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1] and ed_matrix[i][j] == ed_matrix[i - 1][j - 1]:
            i -= 1
            j -= 1
            continue
        if i > 0 and j > 0 and ed_matrix[i][j] == ed_matrix[i - 1][j - 1] + 1:
            sub_dict[f"{s1[i - 1]}{s2[j - 1]}"] = sub_dict.get(f"{s1[i - 1]}{s2[j - 1]}", 0) + 1
            i -= 1
            j -= 1
            continue
        # del?
        if i > 0 and ed_matrix[i][j] == ed_matrix[i - 1][j] + 1:
            if j > 0:
                ins_dict[f"{s2[j - 1]}{s1[i - 1]}"] = ins_dict.get(f"{s2[j - 1]}{s1[i - 1]}", 0) + 1
            else:
                ins_dict[f"<s>{s1[i - 1]}"] = ins_dict.get(f"<s>{s1[i - 1]}", 0) + 1
            i -= 1
            continue
        # ins?
        if j > 0 and ed_matrix[i][j] == ed_matrix[i][j - 1] + 1:
            # if we need to insert a char into s1 to match s2, it means we have deleted a correct char in s2
            if j > 1:
                del_dict[f"{s2[j - 2]}{s2[j - 1]}"] = del_dict.get(f"{s2[j - 2]}{s2[j - 1]}", 0) + 1
            else:
                del_dict[f"<s>{s2[j - 1]}"] = del_dict.get(f"<s>{s2[j - 1]}", 0) + 1
            j -= 1
            continue

    return ins_dict, del_dict, sub_dict, trans_dict


if __name__ == '__main__':
    nltk.data.path.append('/Users/lw/Code/toolkit/nltk_data')
    corpus_f = open('./corpus/reuters.txt', 'r')
    generate_confusion_logp(corpus_f.read())
    corpus_f.close()
