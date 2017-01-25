import nltk
import gensim
import os

TEXTS_DIR = './storage/texts/'
MODELS_DIR = './storage/models'

def iter_thru_doc(topdir, stoplist):
    for textf in os.listdir(topdir):
        with open(os.path.join(topdir, textf), 'r') as tf:
            text = tf.read()
        tf.close()
        yield (x for x in
            gensim.utils.tokenize(text, lowercase=True, deacc=True,
                                 errors="ignore")
            if x not in stoplist)

class MyCorpus(object):
    def __init__(self, topdir, stoplist):
        self.topdir = topdir
        self.stoplist = stoplist
        self.dictionary = gensim.corpora.Dictionary(iter_thru_doc(topdir, stoplist))

    def __iter__(self):
        for tokens in iter_thru_doc(self.topdir, self.stoplist):
            yield self.dictionary.doc2bow(tokens)

def main():
    stoplist = set(nltk.corpus.stopwords.words("english"))
    corpus = MyCorpus(TEXTS_DIR, stoplist)
    corpus.dictionary.save(os.path.join(MODELS_DIR, "mtsamples.mm"))
    gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "mtsamples.mm"),
                                      corpus)



if __name__ == '__main__':
    main()
