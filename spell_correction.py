import string
import nltk
import math

nltk.data.path.append('/Users/lw/Code/toolkit/nltk_data')

from language_model import NgramModel
from channel_model import ConfusionMatrix, ChannelModel
from nltk.tokenize import word_tokenize


class SpellChecker:
    def __init__(self,
                 vocab_path='./vocab.txt',
                 unigram_model_path='./models/reuters_unigram_logp',
                 bigram_model_path='./models/reuters_bigram_logp',
                 trigram_model_path='./models/reuters_trigram_logp'):
        self.lm = NgramModel(unigram_model_path, bigram_model_path, trigram_model_path)
        confusion_from_paper = ConfusionMatrix()
        self.cm = ChannelModel(vocab_path, confusion_from_paper)
        self.non_word_max_ed = 2  # Maximum of edit distance (the higher, the better)
        self.real_word_max_ed = 1  # recommended 1

    def check_unigram(self, sent: str):
        sent, _ = self.__check_non_word(sent, (lambda _, __, sub_word: self.lm.logp_unigram(sub_word)))

        return sent

    def check_bigram(self, sent: str):
        sent, non_word_count = self.__check_non_word(sent, self.lm.logp_word_in_sent_bigram)
        sent, _ = self.__check_real_word(sent, self.lm.logp_word_in_sent_bigram)

        return sent

    def check_trigram(self, sent: str):
        sent, non_word_count = self.__check_non_word(sent, self.lm.logp_word_in_sent_trigram)
        sent, _ = self.__check_real_word(sent, self.lm.logp_word_in_sent_trigram)

        return sent

    def __check_non_word(self, sent: str, lm_method):
        words = word_tokenize(sent)
        non_words = self.__find_non_word(sent)

        if len(non_words) == 0:
            return sent, 0

        # Get Candidates
        for non_word, non_word_index in non_words:
            ed = 1
            candidates = self.cm.get_candidates(non_word, edit_distance=ed)
            while ed < self.non_word_max_ed or len(candidates) == 0:  # Make sure every non_word will be substitute
                ed += 1
                new_candidates = self.cm.get_candidates(non_word, edit_distance=ed)
                # Update candidates
                for key, value in new_candidates.items():
                    if key in candidates:
                        candidates[key] = math.log10(pow(10, candidates[key]) + pow(10, value))
                    else:
                        candidates[key] = value
            if len(candidates) == 0:
                return sent, 0

            # Search for max prob candidate
            max_logprob_cand = non_word
            max_logprob = -math.inf
            for cand_word in candidates.keys():
                noisy_logprob = candidates[cand_word]
                lm_logprob = lm_method(words, non_word_index, cand_word)
                logprob = noisy_logprob + lm_logprob
                if logprob > max_logprob:
                    max_logprob = logprob
                    max_logprob_cand = cand_word

            # Substitute the non_word
            sent = self.__substitue_word(sent, words, non_word_index, max_logprob_cand)
            words[non_word_index] = max_logprob_cand

        return sent, len(non_words)

    def __check_real_word(self, sent: str, lm_method):
        words = word_tokenize(sent)
        correct_count = 0

        for i in range(len(words)):
            real_word = words[i]

            # Get Candidates
            ed = 1
            candidates = self.cm.get_candidates(real_word, edit_distance=ed)
            # while len(candidates) == 0 and ed < self.max_ed:
            while ed < self.real_word_max_ed:
                ed += 1
                new_candidates = self.cm.get_candidates(real_word, edit_distance=ed)
                # Update candidates
                for key, value in new_candidates.items():
                    if key in candidates:
                        candidates[key] = math.log10(pow(10, candidates[key]) + pow(10, value))
                    else:
                        candidates[key] = value
            if len(candidates) == 0:
                continue
            candidates[real_word] = 0  # Correction of real_word noisy prob

            max_logprob_cand = real_word
            max_logprob = -math.inf
            for cand_word in candidates.keys():
                noisy_logprob = candidates[cand_word]
                lm_logprob = lm_method(words, i, cand_word)
                logprob = noisy_logprob + lm_logprob
                # logprob = lm_logprob
                if logprob > max_logprob:
                    max_logprob = logprob
                    max_logprob_cand = cand_word

            # Substitute the real_word
            if max_logprob_cand != real_word:
                sent = self.__substitue_word(sent, words, i, max_logprob_cand)
                words[i] = max_logprob_cand
                correct_count += 1

        return sent, correct_count

    def __find_non_word(self, sent: str):
        words = word_tokenize(sent)  # preprocess
        non_words = []
        for i in range(len(words)):
            word = words[i]
            if not self.cm.search_word(word):
                non_words.append((word, i))  # word and the index in tokenized words
        return non_words

    def __substitue_word(self, sent: str, words: list, oword_index, sub_word):
        oword = words[oword_index]

        # 逐跳搜索方法，确保找到的是位置正确的词，而不是其他位置的相同词
        start_index = 0
        for i in range(oword_index):
            word_start = sent.find(words[i], start_index)
            assert word_start != -1
            start_index = word_start + len(words[i])
        start_index = sent.find(oword, start_index)
        assert start_index != -1

        # 替换
        sent = sent[: start_index] + sub_word + sent[start_index + len(oword):]

        return sent
