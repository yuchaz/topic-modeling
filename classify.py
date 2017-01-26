from packages.extract_texts import extract_all_jtpair
from packages.k_means import run_k_mean_and_get_optimal_k, run_k_mean_with_k
import gensim

DICT_DIR = './storage/models/text_corpus.dict'
CORPUS_DIR = './storage/models/text_corpus.mm'
DATA_DIR = './scienceie2017_data/train/'


def main():
    dictionary = gensim.corpora.Dictionary.load(DICT_DIR)
    corpus = gensim.corpora.MmCorpus(CORPUS_DIR)

    tfidf = gensim.models.TfidfModel(corpus, normalize=True)
    corpus_tfidf = tfidf[corpus]

    lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=30)
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
