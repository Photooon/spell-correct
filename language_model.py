import json
from math import pow, log10


class NgramModel:
    def __init__(self, unigram_path, bigram_path, trigram_path):
        self.unigram = {}
        self.bigram = {}
        self.trigram = {}
        self.__load_from_json_file(unigram_path, bigram_path, trigram_path)

    def __load_from_json_file(self, unigram_path, bigram_path, trigram_path):
        self.unigram = json.load(open(unigram_path, 'r'))
        self.bigram = json.load(open(bigram_path, 'r'))
        self.trigram = json.load(open(trigram_path, 'r'))

    def __load_from_file(self, model_path):
        """
        load from srilm lm
        :param model_path:
        :return:
        """
        with open(model_path, 'r') as f:
            raw_str = f.read()
            raw_blocks = raw_str.split('\\')
            count_str = raw_blocks[2]
            unigram_str = raw_blocks[3]
            bigram_str = raw_blocks[4]
            trigram_str = raw_blocks[5]

            # process count_str
            lines = count_str.split('\n')
            unigram_count_str = lines[1]
            parts = unigram_count_str.split('=')
            self.unigram_count = int(parts[1])
            bigram_count_str = lines[2]
            parts = bigram_count_str.split('=')
            self.bigram_count = int(parts[1])
            trigram_count_str = lines[3]
            parts = trigram_count_str.split('=')
            self.trigram_count = int(parts[1])

            # process unigram_str
            for line in unigram_str.split('\n'):
                parts = line.split('\t')
                if len(parts) == 1:  # skip the title
                    continue
                elif len(parts) == 2:  # log(p), word
                    self.unigram[f"{parts[1]}"] = float(parts[0])
                elif len(parts) == 3:  # log(p), word, log(backoff)
                    self.unigram[f"{parts[1]}"] = float(parts[0])
                else:  # unknown
                    continue

            # process bigram_str
            for line in bigram_str.split('\n'):
                parts = line.split('\t')
                if len(parts) == 1:  # skip the title
                    continue
                elif len(parts) == 2 or len(parts) == 3:
                    words = parts[1].split(' ')
                    if len(words) == 2:  # log(p), word1, word2
                        self.bigram[f"{words[0]}\t{words[1]}"] = float(parts[0])
                    else:  # unknown
                        continue
                else:  # unknown
                    continue

            # process trigram_str
            for line in trigram_str.split('\n'):
                parts = line.split('\t')
                if len(parts) == 1:
                    continue
                elif len(parts) == 2 or len(parts) == 3:
                    words = parts[1].split(' ')
                    if len(words) == 3:
                        self.trigram[f"{words[0]}\t{words[1]}\t{words[2]}"] = float(parts[0])
                    else:
                        continue
                else:
                    continue

    def logp_unigram(self, w1):
        if w1 in self.unigram:
            return self.unigram[f"{w1}"]
        else:
            return self.unigram['<unk>']                # OOV

    def logp_bigram(self, w1, w2):
        if f"{w1}\t{w2}" in self.bigram:
            return self.bigram[f"{w1}\t{w2}"]
        else:
            return log10(0.4 * self.p_unigram(w2))      # backoff

    def logp_trigram(self, w1, w2, w3):
        if f"{w1}\t{w2}\t{w3}" in self.trigram:
            return self.trigram[f"{w1}\t{w2}\t{w3}"]
        else:
            return log10(0.4 * self.p_bigram(w2, w3))   # backoff

    def logp_interpolation(self, w1, w2):
        pass

    def p_unigram(self, w1):
        return pow(10, self.logp_unigram(w1))

    def p_bigram(self, w1, w2):
        return pow(10, self.logp_bigram(w1, w2))

    def p_trigram(self, w1, w2, w3):
        return pow(10, self.logp_trigram(w1, w2, w3))

    def logp_word_in_sent_bigram(self, words, word_index, sub_word):
        prob = 0
        if word_index > 0:
            prob += self.logp_bigram(words[word_index - 1], sub_word)
        if word_index < len(words) - 1:
            prob += self.logp_bigram(sub_word, words[word_index + 1])
        return prob

    def logp_word_in_sent_trigram(self, words, word_index, sub_word):
        prob = 0
        if word_index > 1:
            prob += self.logp_trigram(words[word_index - 2], words[word_index - 1], sub_word)
        if 0 < word_index < len(words) - 1:
            prob += self.logp_trigram(words[word_index - 1],  sub_word, words[word_index + 1])
        if word_index < len(words) - 2:
            prob += self.logp_trigram(sub_word, words[word_index + 1], words[word_index + 2])
        return prob
