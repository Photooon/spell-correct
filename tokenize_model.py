import nltk


def word_tokenize(sent: str):
    nltk.data.path.append('/Users/lw/Code/toolkit/nltk_data')
    words = nltk.word_tokenize(sent)
    # words = [word_trim(word) for word in sent.split(' ') if word]   # split and remove the empty string
    return words


def word_trim(word):
    # remove the front punctuation
    while len(word) > 1 and word[0] in string.punctuation:
        word = word[1:]

    # remove the back punctuation
    while len(word) > 1 and word[-1] in string.punctuation:
        if word[-1] == '\'':
            break
        word = word[:-1]
    return word
