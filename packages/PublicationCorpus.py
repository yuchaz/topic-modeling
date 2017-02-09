import gensim
import os
import unicodedata as ud
import itertools
from collections import defaultdict
import operator
import math
import nltk
import packages.data_path_parser as dp


filename_list = ['CSE','physics','MSE']
NUM_CLASS = len(filename_list)

datapath_dict = {
    'train': dp.get_annotated_training_set(),
    'dev': dp.get_annotated_dev_set(),
    'test': dp.get_annotated_test_set()
}

def get_stoplist():
    return set(nltk.corpus.stopwords.words("english"))

def get_home_paths_from_tags(*tags):
    homedirs = [datapath_dict.get(tag) for tag in list(tags)]
    if len(homedirs) == 0:
        raise ValueError('You should only input train, dev or test set')
    else:
        return homedirs

class Corpus(object):
    def __init__(self,*homedirs):
        self.homedirs = list(homedirs)
        self.stoplist = get_stoplist()
        self.journal_categories = []

    def __iter__(self):
        all_corpus = itertools.chain(*[extract_from_texts(homedir, self.stoplist) for homedir in self.homedirs])
        for tokens, journal_category in all_corpus:
            self.journal_categories.append(journal_category)
            yield self.dictionary.doc2bow(tokens)

class EvaluationCorpus(Corpus):
    def __init__(self,dictionary,*tags):
        homedirs = get_home_paths_from_tags(*tags)
        Corpus.__init__(self,*homedirs)
        self.dictionary = dictionary

class TrainingCorpus(Corpus):
    def __init__(self,informative_amount,*tags):
        homedirs = get_home_paths_from_tags(*tags)
        Corpus.__init__(self,*homedirs)
        self.dictionary = gensim.corpora.Dictionary(extract_for_dict(self.homedirs, self.stoplist))
        if informative_amount != 0:
            self.dictionary.filter_tokens(good_ids=self.get_informative_terms(informative_amount))

    def get_informative_terms(self,feature_amount_each):
        all_corpus = itertools.chain(*[extract_from_texts(homedir, self.stoplist) for homedir in self.homedirs])
        dict_lst = [defaultdict(lambda:1.0) for i in range(NUM_CLASS)]
        mi_lst = [defaultdict(lambda:1.0) for i in range(NUM_CLASS)]
        term_list = self.dictionary.keys()
        for doc, journal_category in all_corpus:
            for term, fqcy in self.dictionary.doc2bow(doc):
                dict_lst[int(journal_category)][term] += 1

        term_count = {term: sum(dict_lst[jt][term] for jt in range(NUM_CLASS)) for term in term_list}
        class_count = {clss: sum(dict_lst[clss][term] for term in term_list) for clss in range(NUM_CLASS)}
        total_count = sum(v for k,v in class_count.items())

        for category in range(NUM_CLASS):
            for term in term_list:
                n11 = dict_lst[category][term]
                mi_lst[category][term] = calc_mutual_information(n11,
                                                class_count[category],
                                                term_count[term],
                                                total_count)

        sorted_mi = [[]]*NUM_CLASS
        for category in range(NUM_CLASS):
            sorted_mi[category] = sorted(mi_lst[category].items(), key=operator.itemgetter(1), reverse=True)

        good_ids = []
        for category in range(NUM_CLASS):
            sorted_mi_idlist = [mi[0] for mi in sorted_mi[category]]
            with open('./storage/{}.txt'.format(filename_list[category]), 'w+') as fn:
                for mi in sorted_mi[category][:feature_amount_each]:
                    fn.write('{}\t{}\t{}\n'.format(mi[0], self.dictionary.get(mi[0]), mi[1]))
            fn.close()

            good_ids.extend(sorted_mi_idlist[:feature_amount_each])
        return set(good_ids)

def calc_mutual_information(n11,n_1,n1_,n_doc):
    n10 = n1_ - n11
    n01 = n_1 - n11
    n00 = n_doc - n11 - n10 - n01
    return n11/n_doc * math.log(n11*n_doc / (n_1*n1_),2) + \
           n01/n_doc * math.log(n01*n_doc / ((n01+n00)*n_1),2) + \
           n10/n_doc * math.log(n10*n_doc / ((n10+n00)*n1_),2) + \
           n00/n_doc * math.log(n00*n_doc / ((n01+n00)*(n10+n00)),2)

def extract_for_dict(homedirs, stoplist):
    for tokens, jn in itertools.chain(*[extract_from_texts(homedir, stoplist) for homedir in homedirs]):
        yield tokens

def extract_from_texts(homedir, stoplist):
    for text_name in os.listdir(homedir):
        if text_name == '.gitkeep': continue

        with open(os.path.join(homedir, text_name)) as text_file:
            texts_raw = text_file.read()
        text_file.close()
        texts, journal_category = texts_raw.split('\t\t\t')
        yield [to_lower(token)
            for token in gensim.utils.tokenize(texts, deacc=True, errors="ignore")
            if to_lower(token) not in stoplist], journal_category

def to_lower(token):
    return token.lower() if sum(1 for c in token if ud.category(c)=='Lu')==1 else token
