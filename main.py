from packages.PublicationCorpus import PublicationCorpus
import os
import nltk
import gensim

TEXT_DIR = './storage/texts/'
JT_DIR= './storage/journalname_title/'
MODELS_DIR = './storage/models'
DATA_DIR = './scienceie2017_data/train/'

def main():
    stoplist = set(nltk.corpus.stopwords.words("english"))
    journal_stoplist = set('letters journal annals international current opinion equilibria fig eq'.split())
    stoplist.update(journal_stoplist)

    text_corpus = PublicationCorpus(TEXT_DIR, stoplist)
    jt_corpus = PublicationCorpus(JT_DIR, stoplist)

    text_corpus.dictionary.save(os.path.join(MODELS_DIR, "text_corpus.dict"))
    gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "text_corpus.mm"),
                                      text_corpus)

    jt_corpus.dictionary.save(os.path.join(MODELS_DIR, "jt_corpus.dict"))
    gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "jt_corpus.mm"),
                                      jt_corpus)

if __name__ == '__main__':
    main()
