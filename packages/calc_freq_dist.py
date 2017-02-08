from collections import defaultdict
import os


def calc_freq_dist(homedir):
    dictionary = defaultdict(int)
    for fn in os.listdir(homedir):
        with open(os.path.join(homedir,fn)) as fnnmae:
            full_text = fn.read()
        fnnmae.close()
    for fn in os.listdir(homedir):
        with open()
        full_text = fn.read()

    for token in token_list:
        dictionary[token] += 1
    return dictionary
