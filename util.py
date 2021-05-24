import nltk


def clean_corpus(corpus_str, corpus_path):
    sents = nltk.tokenize.sent_tokenize(corpus_str)
    f = open(corpus_path, 'w')
    for sent in sents:
        words = nltk.tokenize.word_tokenize(sent)
        words = ['<s>'] + words + ['</s>']
        words_str = ' '.join(words)
        f.write(words_str)
    f.close()


if __name__ == '__main__':
    nltk.data.path.append('/Users/lw/Code/toolkit/nltk_data')
    reuters = nltk.corpus.reuters
    clean_corpus(reuters.raw(), './corpus/reuters.txt')
