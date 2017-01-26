from packages.PublicationCorpus import PublicationCorpus
from packages.extract_texts import extract_all_jtpair
from packages.k_means import run_k_mean_and_get_optimal_k, run_k_mean_with_k
import os
import nltk
import gensim
import pprint

HOMEDIR = './storage/texts/'
# HOMEDIR = './storage/journalname_title/'
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
    lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=30)
    # transformation from tfidf to lsi
    corpus_lsi = lsi[corpus_tfidf]
    optimal_clusters = run_k_mean_and_get_optimal_k(corpus_lsi, 30)
    lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=optimal_clusters)
    corpus_lda = lda[corpus]
    labels = run_k_mean_with_k(corpus_lda, optimal_clusters)


    max_l = [max(doc, key=lambda i: abs(i[1])) for doc in corpus_lda]
    idx = 0
    for jname in extract_all_jtpair(DATA_DIR):
        # print "{}\t{}".format(max_l[idx][0], jname)
        print "{}\t{}".format(labels[idx], jname)
        idx += 1

if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    main()
