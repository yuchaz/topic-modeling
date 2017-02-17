import gensim
import os
import unicodedata as ud
import itertools
from collections import defaultdict
import operator
import math
import nltk
import packages.data_path_parser as dp
import re
from packages.extract_texts import extract_journalname_from_xml
import packages.data_path_parser as dp

filename_list = ['CSE','physics','MSE']
NUM_CLASS = len(filename_list)

datapath_dict = {
    'train': dp.get_annotated_training_set(),
    'dev': dp.get_annotated_dev_set(),
    'test': dp.get_annotated_test_set()
}

strange_dict = {
    './storage/texts/train': dp.get_training_corpus(),
    './storage/texts/dev': dp.get_dev_corpus()
}

def get_stoplist():
    handcraft_stopwords = """letter BY elsevier responsibility license com
                             procedia sciencedirect ND NC CC IERI ieri abstract
                             online institute tel conference mail keywords doi
                             open selection ScienceDirect creativecommons licenses
                             published http procedia www peer acknowledgements
                             journal review international references AASRI aasri
                             IEEE org fax""".split()
    stopwords = set(nltk.corpus.stopwords.words("english")+handcraft_stopwords)
    return stopwords

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
        self.journal_name = []

    def __iter__(self):
        all_corpus = itertools.chain(*[extract_from_texts(homedir, self.stoplist) for homedir in self.homedirs])
        for tokens, journal_category, journal_name in all_corpus:
            self.journal_categories.append(journal_category)
            self.journal_name.append(journal_name)
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
        # mi_lst = [defaultdict(lambda:1.0) for i in range(NUM_CLASS)]
        mi_lst = defaultdict(lambda:1.0)
        term_list = self.dictionary.keys()
        for doc, journal_category, jn in all_corpus:
            for term, fqcy in self.dictionary.doc2bow(doc):
                dict_lst[int(journal_category)][term] += 1

        term_count = {term: sum(dict_lst[jt][term] for jt in range(NUM_CLASS)) for term in term_list}
        class_count = {clss: sum(dict_lst[clss][term] for term in term_list) for clss in range(NUM_CLASS)}
        total_count = sum(v for k,v in class_count.items())

        for term in term_list:
            mi_lst[term] = sum((calc_mutual_information(dict_lst[category][term],
                                            class_count[category],
                                            term_count[term],
                                            total_count) for category in range(NUM_CLASS)))

        sorted_mi = sorted(mi_lst.items(), key=operator.itemgetter(1), reverse=True)
        # for category in range(NUM_CLASS):
        #     sorted_mi[category] = sorted(mi_lst[category].items(), key=operator.itemgetter(1), reverse=True)

        # good_ids = []
        # for category in range(NUM_CLASS):
        #     sorted_mi_idlist = [mi[0] for mi in sorted_mi[category]]
        #     with open('./storage/{}.txt'.format(filename_list[category]), 'w+') as fn:
        #         for mi in sorted_mi[category][:feature_amount_each]:
        #             fn.write('{}\t{}\t{}\n'.format(mi[0], self.dictionary.get(mi[0]), mi[1]))
        #     fn.close()
        #
        #     good_ids.extend(sorted_mi_idlist[:feature_amount_each])
        good_ids = [mi[0] for mi in sorted_mi[:feature_amount_each*3]]
        with open('./storage/mi.txt', 'w+') as fn:
            for mi in sorted_mi[:feature_amount_each*3]:
                fn.write('{}\t{}\t{}\n'.format(mi[0], self.dictionary.get(mi[0]), mi[1]))
            fn.close()
        return set(good_ids)

def calc_mutual_information(n11,n_1,n1_,n_doc):
    n10 = n1_ - n11
    n01 = n_1 - n11
    n00 = n_doc - n11 - n10 - n01
    return n11/n_doc * math.log(n11*n_doc / (n_1*n1_),2) + \
           n01/n_doc * math.log(n01*n_doc / ((n01+n00)*n_1),2)

    # return n11/n_doc * math.log(n11*n_doc / (n_1*n1_),2) + \
    #        n01/n_doc * math.log(n01*n_doc / ((n01+n00)*n_1),2) + \
    #        n10/n_doc * math.log(n10*n_doc / ((n10+n00)*n1_),2) + \
    #        n00/n_doc * math.log(n00*n_doc / ((n01+n00)*(n10+n00)),2)

def extract_for_dict(homedirs, stoplist):
    for tokens, jn, jn_name in itertools.chain(*[extract_from_texts(homedir, stoplist) for homedir in homedirs]):
        yield tokens

def extract_from_texts(homedir, stoplist):
    for text_name in os.listdir(homedir):
        if text_name == '.gitkeep': continue

        with open(os.path.join(homedir, text_name)) as text_file:
            texts_raw = text_file.read()
        text_file.close()
        texts, journal_category, journal_name = texts_raw.split('\t\t\t')
        # write_error_journal_name(text_name,homedir,journal_category,texts)
        yield [to_lower(token)
            for token in gensim.utils.tokenize(texts, deacc=True, errors="ignore")
            if to_lower(token) not in stoplist], journal_category, journal_name

def to_lower(token):
    return token.lower() if sum(1 for c in token if ud.category(c)=='Lu')==1 else token

def write_error_journal_name(text_name,homedir,journal_category,texts):
    list_to_find = 'temperatures temperature electron quantum scattering'.split()
    rgx = re.compile('|'.join(map(re.escape, list_to_find)))
    if int(journal_category) == 0 and rgx.search(texts):
        with open ('./error_journals.txt', 'a') as errfile:
            errfile.write('{}\t{}\n'.format(text_name,
                extract_journalname_from_xml(os.path.join(strange_dict[homedir], os.path.splitext(text_name)[0]+'.xml'))))
        errfile.close()
