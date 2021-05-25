import json
from math import log10


def generate_count_file(corpus_path: str):
    corpus_f = open(corpus_path, 'r')
    sents = [line.rstrip() for line in corpus_f.readlines()]
    corpus_f.close()

    unigram_count = {}
    bigram_count = {}
    trigram_count = {}

    for sent in sents:
        words = sent.split(' ')

        for i in range(len(words)):
            word = words[i]
            # add into unigram
            unigram_count[f"{word}"] = unigram_count.get(f"{word}", 0) + 1
            # add into bigram
            if i < len(words) - 1:
                bigram_count[f"{word}\t{words[i + 1]}"] = bigram_count.get(f"{word}\t{words[i + 1]}", 0) + 1
            # add into trigram
            if i < len(words) - 2:
                trigram_count[f"{word}\t{words[i + 1]}\t{words[i + 2]}"] = trigram_count.get(f"{word}\t{words[i + 1]}\t{words[i + 2]}", 0) + 1

    return unigram_count, bigram_count, trigram_count


def generate_origin_model(unigram_count: dict, bigram_count: dict, trigram_count: dict):
    pass


def generate_addsmooth_model(k: float, unigram_count: dict, bigram_count: dict, trigram_count: dict):
    unigram_logp = {}
    bigram_logp = {}
    trigram_logp = {}

    V = len(unigram_count.keys())
    uni_total_count = 0
    for value in unigram_count.values():
        uni_total_count += value

    # unigram
    for key, value in unigram_count.items():
        unigram_logp[key] = log10((value + k) / (uni_total_count + V * k))
    unigram_logp['<unk>'] = log10(k / (uni_total_count + V * k))

    # bigram
    for key, value in bigram_count.items():
        words = key.split('\t')
        bigram_logp[key] = log10((value + k) / (unigram_count[words[0]] + V * k))

    # trigram
    for key, value in trigram_count.items():
        words = key.split('\t')
        trigram_logp[key] = log10((value + k) / (bigram_count[f"{words[0]}\t{words[1]}"] + V * k))

    return unigram_logp, bigram_logp, trigram_logp


def generate_goodturing_model(count_file_path, model_path='./model/goodturing_model.txt'):
    pass


def generate_kndiscount_model(count_file_path, model_path='./model/kndiscount_model.txt'):
    pass


if __name__ == '__main__':
    unigram_count, bigram_count, trigram_count = generate_count_file('./corpus/reuters.txt')

    # add-k smoothing
    k = 0.000001
    unigram_logp, bigram_logp, trigram_logp = generate_addsmooth_model(k, unigram_count, bigram_count, trigram_count)

    # Save
    json.dump(unigram_logp, open('./models/reuters_unigram_logp', 'w'))
    json.dump(bigram_logp, open('./models/reuters_bigram_logp', 'w'))
    json.dump(trigram_logp, open('./models/reuters_trigram_logp', 'w'))
