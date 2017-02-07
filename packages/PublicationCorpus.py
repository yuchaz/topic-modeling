import gensim
import os
import unicodedata as ud
import itertools
from collections import defaultdict
import operator
import math

class TestCorpus(object):
    def __init__(self, homedir, stoplist, dictionary):
        self.homedir = homedir
        self.stoplist = stoplist
        self.dictionary = dictionary
        self.journal_categories = []

    def __iter__(self):
        for tokens, journal_category in extract_from_texts(self.homedir, self.stoplist):
            self.journal_categories.append(journal_category)
            yield self.dictionary.doc2bow(tokens)

class PublicationCorpus(object):
    def __init__(self, stoplist, *homedirs):
        self.homedirs = list(homedirs)
        self.stoplist = stoplist
        self.dictionary = gensim.corpora.Dictionary(extract_for_dict(self.homedirs, self.stoplist))
        self.journal_categories = []
        self.dictionary.filter_tokens(good_ids=self.get_informative_terms())

    def __iter__(self):
        all_corpus = itertools.chain(*[extract_from_texts(homedir, self.stoplist) for homedir in self.homedirs])
        for tokens, journal_category in all_corpus:
            self.journal_categories.append(journal_category)
            yield self.dictionary.doc2bow(tokens)

    def get_informative_terms(self):
        all_corpus = itertools.chain(*[extract_from_texts(homedir, self.stoplist) for homedir in self.homedirs])
        dict_lst = [defaultdict(lambda:1.0) for i in range(3)]
        mi_lst = [defaultdict(lambda:1.0) for i in range(3)]
        term_list = self.dictionary.keys()
        for doc, journal_category in all_corpus:
            for term in term_list:
                if self.dictionary.get(term) in doc:
                    dict_lst[int(journal_category)][term] += 1

        term_count = {term: sum(dict_lst[jt][term] for jt in range(3)) for term in term_list}
        class_count = {clss: sum(dict_lst[clss][term] for term in term_list) for clss in range(3)}
        total_count = sum(v for k,v in class_count.items())

        for category in range(3):
            for term in term_list:
                n11 = dict_lst[category][term]
                mi_lst[category][term] = calc_mutual_information(n11,
                                                class_count[category],
                                                term_count[term],
                                                total_count)

        sorted_mi = [[]]*3
        for category in range(3):
            sorted_mi[category] = sorted(mi_lst[category].items(), key=operator.itemgetter(1))

        good_ids = []
        for category in range(3):
            sorted_mi_idlist = [mi[0] for mi in sorted_mi[category]]
            good_ids.extend(sorted_mi_idlist[:1500])
        return set(good_ids)

def calc_mutual_information(n11,n_1,n1_,n_doc):
    n10 = n1_ - n11
    n01 = n_1 - n11
    n00 = n_doc - n11 - n10 - n01
    return n11/n_doc * math.log(n11*n_doc / (n_1*n1_),2) + \
           n01/n_doc * math.log(n01*n_doc / ((n01+n00)*n_1),2) + \
           n10/n_doc * math.log(n10*n_doc / ((n10+n00)*n1_),2) + \
           n00/n_doc * math.log(n00*n_doc / ((n01+n00)*(n10+n00)),2)

def extract_from_texts(homedir, stoplist):
    for text_name in os.listdir(homedir):
        if text_name == '.gitkeep': continue

        with open(os.path.join(homedir, text_name)) as text_file:
            texts_raw = text_file.read()
        text_file.close()
        texts, journal_category = texts_raw.split('\t\t\t')
        yield [token.lower() if sum(1 for c in token if ud.category(c)=='Lu')==1 else token for token in
            gensim.utils.tokenize(texts, deacc=True, errors="ignore")
            if token not in stoplist], journal_category

def extract_for_dict(homedirs, stoplist):
    for tokens, jn in itertools.chain(*[extract_from_texts(homedir, stoplist) for homedir in homedirs]):
        yield tokens

class SklearnCorpus(object):
    def __init__(self, homedir):
        self.homedir = homedir
        self.journal_categories = []

    def __iter__(self):
        for tokens, journal_category in sklearn_extract_from_texts(self.homedir):
            self.journal_categories.append(journal_category)
            yield tokens

def sklearn_extract_from_texts(homedir):
    for text_name in os.listdir(homedir):
        if text_name == '.gitkeep': continue

        with open(os.path.join(homedir, text_name)) as text_file:
            texts_raw = text_file.read()
        text_file.close()
        texts, journal_category = texts_raw.split('\t\t\t')
        yield texts, journal_category
