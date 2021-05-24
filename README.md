# I. Dataset

- DeepMind Q&A Dataset
  - Content: CNN articles, Daily Mail articles
  - Reference: https://cs.nyu.edu/~kcho/DMQA/



# II. Preprocess

fetch data:

```python
import nltk

nltk.data.path.append('/Users/lw/Code/toolkit/nltk_data')
raw = nltk.corpus.reuters.raw()
with open('./corpus/reuters_raw.txt', 'w') as f:
    f.write(raw)
```

generate models:

```sh
./ngram-count -read ../corpus/reuters_train_countfile -lm ../corpus/reuters_train_kndiscount_lm -interpolate -kndiscount
```

```sh
./ngram-count -read ../corpus/reuters_train_countfile -lm ../corpus/reuters_train_addone_lm -interpolate -addsmooth
```

```sh
./ngram-count -read ../corpus/reuters_train_countfile -lm ../corpus/reuters_train_goodturing_lm
```

ppl test:

```sh
./ngram -lm ../models/reuters_train_addone_lm -ppl ../corpus/reuters_test_raw
```

test result:

| model                  | logprob   | ppl      | ppl1     |
| ---------------------- | --------- | -------- | -------- |
| reusters_goodturing_lm | -938098.7 | 251.027  | 525.1339 |
| reusters_addone_lm     |           |          |          |
| reusters_kndiscount_lm | -926511.2 | 234.4655 | 486.0366 |



# III. Language model

- Strategy: add-k smooth and stupid backoff

A big problem:

srilm will backoff to (n-1)-gram if the count of n-gram is zero, and do not use the add-k smooth. Maybe the reason is that, if we use the add-k smooth method for every combination, the file is too large to save the probability.

reference:

- https://zhuanlan.zhihu.com/p/99906900 (a big problem)

# IV. Channel model

## Candidate model

use the trie as basic data structure.

we can get candidates of a word from model by dfs the trie.

a problem: 只乘以修改部分的概率吗？

reference:

- https://www.geeksforgeeks.org/spell-checker-using-trie/

## Confusion Matrix

use Damerau-Levenshtein distance

generate matrix from spell-error.txt(for error count) and corpus of reuters(for char statistics)

smooth with add one method.

a problem: 为什么用错误的数量除以出现的频率

此外还有大小写的问题？

reference: https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance

- i used the matrix from paper for comparison, and substitue the ' with " and # with <s> in the data file.

# V. Experiment Result

non-word only:

| corpus  | smoothing method | language model | channel model | accuracy(%) | time(s) |
| ------- | ---------------- | -------------- | ------------- | ----------- | ------- |
| reuters | add-k            | unigram        | -             | 85.90       | 3.377   |
| reuters | add-k            | bigram         | -             | 85.7        | 3.474   |
| reuters | add-k            | trigram        | -             | 84.89       | 3.329   |



# VI. Some Thoughs

- Lowercase and Uppercase Problem!
- Punctuation Problem!
- make OOVs



# Summary

- 我使用了noisy_logprob + lm_logprob的方式，减少了real word误纠的问题
- 我预先处理了数据集
- 我发现使用极小的k能减少对lm的影响
- 我没有处理标点符号的问题
- 对confusion_matrix，大小写统一使用小写
- 将不同路径叠加
- 使用广义编辑距离，人们对于字符的交换操作频率远高于插入和删除（与键盘有关，输入容易交错）。如果对替换操作之后的子串，仍允许操作，可以发现能更好的反映noisy prob。