from packages.PublicationCorpus import PublicationCorpus
from packages.extract_texts import extract_all_jtpair
import os
import nltk
import gensim
import pprint

# HOMEDIR = './storage/texts/'
HOMEDIR = './storage/journalname_title/'
MODELS_DIR = './storage/models'
DATA_DIR = './scienceie2017_data/train/'

def main():
    stoplist = set(nltk.corpus.stopwords.words("english"))
    journal_stoplist = set('letters journal annals international current opinion equilibria'.split())
    stoplist.update(journal_stoplist)
    corpus = PublicationCorpus(HOMEDIR, stoplist)
    corpus.dictionary.save(os.path.join(MODELS_DIR, "pubcorpus.dict"))
    gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "pubcorpus.mm"),
                                      corpus)

    dictionary = corpus.dictionary
    tfidf = gensim.models.TfidfModel(corpus, normalize=True)
    # tfidf is a transformation from bow to tfidf.
    corpus_tfidf = tfidf[corpus]
    lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=3)
    # transformation from tfidf to lsi
    corpus_lsi = lsi[corpus_tfidf]
    max_l = [max(doc, key=lambda i: abs(i[1])) for doc in corpus_lsi]
    idx = 0
    for jname in extract_all_jtpair(DATA_DIR):
        print "{}\t{}".format(max_l[idx][0], jname)
        idx += 1

    print lsi.print_topics()

if __name__ == '__main__':
    main()
