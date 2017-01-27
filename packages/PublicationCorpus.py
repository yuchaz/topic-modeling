import gensim
import os
import unicodedata as ud

class PublicationCorpus(object):
    def __init__(self, homedir, stoplist):
        self.homedir = homedir
        self.stoplist = stoplist
        self.dictionary = gensim.corpora.Dictionary(extract_from_texts(homedir, stoplist))

    def __iter__(self):
        for tokens in extract_from_texts(self.homedir, self.stoplist):
            yield self.dictionary.doc2bow(tokens)

def extract_from_texts(homedir, stoplist):
    for text_name in os.listdir(homedir):
        with open(os.path.join(homedir, text_name)) as text_file:
            texts = text_file.read()
        text_file.close()
        yield (token.lower() if sum(1 for c in token if ud.category(c)=='Lu')==1 else token for token in
            gensim.utils.tokenize(texts, deacc=True, errors="ignore")
            if token not in stoplist and len(token) > 2)
