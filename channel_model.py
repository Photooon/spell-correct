import json
from math import log10


class TrieNode:
    def __init__(self):
        self.children = {}
        self.end = False

    def search(self, word):
        if len(word) == 0:
            return self.end

        p = self
        for ch in word:
            if ch not in p.children:
                return False
            p = p.children[ch]
        if not p.end:
            return False
        return True


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        p = self.root  # 节点指针
        a = p.children
        for ch in word:
            if ch not in p.children:
                p.children[ch] = TrieNode()
            p = p.children[ch]
        p.end = True


class ConfusionMatrix:
    def __init__(self,
                 ins_confusion_path='./confusion_matrix/ins_confusion_matrix.txt',
                 del_confusion_path='./confusion_matrix/del_confusion_matrix.txt',
                 sub_confusion_path='./confusion_matrix/sub_confusion_matrix.txt',
                 trans_confusion_path='./confusion_matrix/trans_confusion_matrix.txt'):
        self.enable = True
        self.ins_dict = json.load(open(ins_confusion_path, 'r'))  # prob(x typed as xy)
        self.del_dict = json.load(open(del_confusion_path, 'r'))  # prob(xy typed as x)
        self.sub_dict = json.load(open(sub_confusion_path, 'r'))  # prob(y typed as x)
        self.trans_dict = json.load(open(trans_confusion_path, 'r'))  # prob(xy typed as yx)

    def logprob_ins(self, x, y):
        x = x.lower()   # Preprocess
        y = y.lower()
        if not self.enable:
            return 0
        x = x if x != '<s>' else '#'
        if (x + y) in self.ins_dict:
            return self.ins_dict[x + y]
        else:
            return self.ins_dict['<unk>']

    def logprob_del(self, x, y):
        x = x.lower()  # Preprocess
        y = y.lower()
        if not self.enable:
            return 0
        x = x if x != '<s>' else '#'
        if (x + y) in self.del_dict:
            return self.del_dict[x + y]
        else:
            return self.del_dict['<unk>']

    def logprob_sub(self, x, y):
        x = x.lower()  # Preprocess
        y = y.lower()
        if not self.enable:
            return 0
        x = x if x != '<s>' else '#'
        if (x + y) in self.sub_dict:
            return self.sub_dict[x + y]
        else:
            return self.sub_dict['<unk>']

    def logprob_trans(self, x, y):
        x = x.lower()  # Preprocess
        y = y.lower()
        if not self.enable:
            return 0
        x = x if x != '<s>' else '#'
        if (x + y) in self.trans_dict:
            return self.trans_dict[x + y]
        else:
            return self.trans_dict['<unk>']

    def prob_ins(self, x, y):
        return pow(10, self.logprob_ins(x, y))

    def prob_del(self, x, y):
        return pow(10, self.logprob_del(x, y))

    def prob_sub(self, x, y):
        return pow(10, self.logprob_del(x, y))

    def prob_trans(self, x, y):
        return pow(10, self.logprob_del(x, y))


class ChannelModel:
    def __init__(self, vocab_path, confusion=None):
        self.trie = Trie()
        self.confusion = ConfusionMatrix() if confusion is None else confusion
        self.candidates = {}
        self.__generate_trie(vocab_path)

    def __generate_trie(self, vocab_path):
        with open(vocab_path) as f:
            words = f.readlines()
            for word in words:
                self.trie.insert(word.rstrip())

    def search_word(self, word):
        return self.trie.root.search(word)

    def get_candidates(self, word, edit_distance=1):
        self.candidates = {}    # clear the cache
        self.__search_candidates(self.trie.root, '', word, 0, edit_distance)
        return self.candidates

    def __search_candidates(self, p: TrieNode, prefix: str, word: str, logprob: float, edit_distance: int):
        # 递归出口
        if edit_distance == 0:
            if p.search(word):
                cand_word = prefix + word
                if cand_word in self.candidates:
                    self.candidates[cand_word] = log10(pow(10, self.candidates[cand_word]) + pow(10, logprob))
                    # different path to the same result should be added
                else:
                    self.candidates[cand_word] = logprob
            return
        elif edit_distance < 0:
            return

        # 递归过程
        # 保持首字母不变
        if len(word) != 0 and word[0] in p.children:
            self.__search_candidates(p.children[word[0]], prefix + word[0], word[1:], logprob, edit_distance)

        # 添加首字母
        for ch in p.children:
            self.__search_candidates(p.children[ch], prefix + ch, word,
                                     logprob + self.confusion.logprob_del(prefix[-1] if len(prefix) > 0 else '<s>', ch),
                                     edit_distance - 1)

        # 删除首字母
        if len(word) > 0:
            self.__search_candidates(p, prefix, word[1:],
                                     logprob + self.confusion.logprob_ins(prefix[-1] if len(prefix) > 0 else '<s>',
                                                                          word[0]),
                                     edit_distance - 1)

        # 置换首字母
        if len(word) > 0:
            for ch in p.children:
                self.__search_candidates(p.children[ch], prefix + ch, word[1:],
                                         logprob + self.confusion.logprob_sub(word[0], ch),
                                         edit_distance - 1)

        # 交换第1、第2个字母
        if len(word) >= 2:
            self.__search_candidates(p, prefix, word[1] + word[0] + word[2:],   # trick: do not add into prefix at first
                                     logprob + self.confusion.logprob_trans(word[1], word[0]),
                                     edit_distance - 1)

        return
