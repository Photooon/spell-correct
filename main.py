from time import time
from spell_correction import SpellChecker


if __name__ == '__main__':
    start_time = time()
    sc = SpellChecker('./vocab.txt')

    testdata_f = open('./testdata.txt', 'r')
    result_f = open('./eval/result.txt', 'w')
    for i in range(1000):
        if i in [33, 267, 364, 497, 677, 730, 955, 970]:
            print("")
        line = testdata_f.readline().split('\t')[2].replace('\n', '')
        # sent = sc.check_unigram(line)
        # sent = sc.check_bigram(line)
        sent = sc.check_trigram(line)
        result_f.write(f"{i + 1}\t{sent}\n")
        print(f"id : {i}\r")
    testdata_f.close()
    result_f.close()

    end_time = time()
    print(f"Time is : {end_time - start_time}s")
