import gensim
import os
import unicodedata as ud

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
    def __init__(self, homedir, stoplist):
        self.homedir = homedir
        self.stoplist = stoplist
        self.dictionary = gensim.corpora.Dictionary(extract_for_dict(homedir, stoplist))
        self.journal_categories = []

    def __iter__(self):
        for tokens, journal_category in extract_from_texts(self.homedir, self.stoplist):
            self.journal_categories.append(journal_category)
            yield self.dictionary.doc2bow(tokens)

def extract_from_texts(homedir, stoplist):
    for text_name in os.listdir(homedir):
        if text_name == '.gitkeep': continue

        with open(os.path.join(homedir, text_name)) as text_file:
            texts_raw = text_file.read()
        text_file.close()
        texts, journal_category = texts_raw.split('\t\t\t')
        yield (token.lower() if sum(1 for c in token if ud.category(c)=='Lu')==1 else token for token in
            gensim.utils.tokenize(texts, deacc=True, errors="ignore")
            if token not in stoplist and len(token) > 2), journal_category

def extract_for_dict(homedir, stoplist):
    for tokens, jn in extract_from_texts(homedir, stoplist):
        yield tokens
