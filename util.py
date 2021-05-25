import nltk


def clean_corpus_str(corpus_str, corpus_path):
    """
    直接使用nltk.corpus.reuters.sents()返回的分词结果错误很多，如'U.S.'分成了'U . S .'
    所以对reuters要半手动分割
    :param corpus_str:
    :param corpus_path:
    :return:
    """
    sents = nltk.tokenize.sent_tokenize(corpus_str)
    f = open(corpus_path, 'w')
    for sent in sents:
        words = nltk.tokenize.word_tokenize(sent)
        words = ['<s>'] + words + ['</s>']
        words_str = ' '.join(words)
        f.write(words_str)
        f.write('\n')
    f.close()


def clean_corpus(sents, corpus_path):
    f = open(corpus_path, 'w')
    for sent in sents:
        words = ['<s>'] + sent + ['</s>']
        words_str = ' '.join(words)
        f.write(words_str)
        f.write('\n')
    f.close()


if __name__ == '__main__':
    nltk.data.path.append('/Users/lw/Code/toolkit/nltk_data')
    reuters = nltk.corpus.reuters
    brown = nltk.corpus.brown
    webtext = nltk.corpus.webtext
    clean_corpus(brown.sents(), './corpus/brown.txt')
    clean_corpus_str(reuters.raw(), './corpus/reuters.txt')
