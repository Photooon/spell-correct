# I. Dataset

- Reuters
- Brown
- Webtext

# II. Preprocess

For each corpus, I tokenize every sentence and add '\<s\>' and '\</s\>' before and after it, and then save them into file.

The Code for this part could be found in "util.py".

# III. Language model

Language model = add-k smoothing + backoff

The model is showed as flow chart follow:

![language_model](https://github.com/Photooon/spell-correct/blob/master/README.assets/language_model.png)

# IV. Channel model

## Candidate model

To enhance the speed of the program, I use Trie to save vocabularies. Each time we want to get candidates for a word, the program will search the Trie with DFS algorithm and Damerau-Levenshtein Distance. The code for this part can be found in "channel_model.py".

Something interesting is that, benifiting from the dfs algorithm, we can search a larger space for candidates compared with dynamic program algorithm. In other words, if we use the dynamic algorithm, the edited characters will not be edited again. But in dfs, we can allow the edited characters to be edited again and again. Therefore, dfs can find more candidates than dynamic program. By the way, I sum the probability of all possible editing path for a candidate word. The code for this part is showed as follow:

```python
def __search_candidates(self, p: TrieNode, prefix, word, logprob, edit_distance):
    # 递归出口
    if edit_distance == 0:
        if p.search(word):
            cand_word = prefix + word
            if cand_word in self.candidates:
                self.candidates[cand_word] = log10(pow(10, self.candidates[cand_word]) + pow(10, logprob))
                # different path to the same result should be added together
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
        self.__search_candidates(p, prefix, word[1] + word[0] + word[2:],
                                     logprob + self.confusion.logprob_trans(word[1], word[0]),
                                     edit_distance - 1)

    return
```

## Confusion Matrix

I generate the confusion matrix from "spell-error.txt". The code for this part can be found in "./confusion_matrix/confusion_matrix.py"

# V. Experiment Result

## Effect of Corpus

| corpus     | sentences count | k     | language model | channel model | accuracy(%) | time(s) |
| ---------- | --------------- | ----- | -------------- | ------------- | ----------- | ------- |
| reuters    | 50981           | 10^-6 | unigram        | √             | 87.6        | 11.3    |
| brown-news | 4623            | 10^-6 | unigram        | √             | 76.9        | 10.2    |
| brown      | 57340           | 10^-6 | unigram        | √             | 79.2        | 11.9    |
| webtext    | 25728           | 10^-6 | unigram        | √             | 70.9        | 10.6    |

We can see that if we use the corpus whose category is close to test set, the result will be more accurate. Moreover, the scale of corpus is also important.

## Effect of k in add-k smoothing

| corpus  | k     | language model | channel model | accuracy(%) | time(s) |
| ------- | ----- | -------------- | ------------- | ----------- | ------- |
| reuters | 10^-7 | trigram        | √             | 99.2        | 18.9    |
| reuters | 10^-6 | trigram        | √             | 99.2        | 18.7    |
| reuters | 10^-5 | trigram        | √             | 99.1        | 18.8    |
| reuters | 1     | trigram        | √             | 64.6        | 19.0    |

From the table we can find that the accuracy increases with the decrease of k. The reason is that if we use large k, such as 1, the probability of words already appeared in corpus will be dominated by the V * k and quickly drop because of the tremendous V(in reuters, V is nearly 60000). And then the differences between candidates will become smaller, making inference more difficult. Laplace smoothing isn't a really good smoothing method. And I suggest that k should be set to (1/ V) if possible.

## Effect of language model

| corpus  | k     | language model | channel model | accuracy(%) | time(s) |
| ------- | ----- | -------------- | ------------- | ----------- | ------- |
| reuters | 10^-6 | unigram        | √             | 87.6        | 11.3    |
| reuters | 10^-6 | bigram         | √             | 96.7        | 18.0    |
| reuters | 10^-6 | trigram        | √             | 99.2        | 18.7    |

This result is reasonable. Something needs to be specified is that the program only check non-word when using unigram model(if we try to check real-word with unigram, it will be a disaster!).

## Effect of channel model

| corpus  | k     | language model | channel model | accuracy(%) | time(s) |
| ------- | ----- | -------------- | ------------- | ----------- | ------- |
| reuters | 10^-6 | trigram        | √             | 99.2        | 18.7    |
| reuters | 10^-6 | trigram        | ×             | 74.7        | 15.3    |

The channel model plays an important role in the spell-correct task.

# VI. Final Model

- Corpus: Reuters
- k: 10^-6
- LM: trigram
- Channel model
- **Accuracy: 99.2%**
- **Time: 0.0187 s/sent**

# VII. Other Things

- When dealing with the real word, I use the noisy_channel_prob * language_model_prob as the probability of the canidate words. And I assume that the noisy_channel_prob of the original word is 1, which makes the change for real word will be difficult and only the highly possible candidate word can be selected to substitute the original word.

# VIII. Reference

- https://www.geeksforgeeks.org/spell-checker-using-trie/
- https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance
- http://norvig.com/ngrams/spell-errors.txt