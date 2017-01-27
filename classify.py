from packages.extract_texts import extract_all_jtpair
from packages.k_means import run_k_mean_and_get_optimal_k, run_k_mean_with_k
import packages.data_path_parser as dp
import gensim
import os

TEXT_DICT_PATH = './storage/models/text_corpus.dict'
TEXT_CORPUS_PATH = './storage/models/text_corpus.mm'
JT_DICT_PATH = './storage/models/jt_corpus.dict'
JT_CORPUS_PATH = './storage/models/jt_corpus.mm'
DATA_DIR = dp.get_training_corpus()
LDA_DIR = './storage/models/'


def main():
    dictionary = gensim.corpora.Dictionary.load(TEXT_DICT_PATH)
    corpus = gensim.corpora.MmCorpus(TEXT_CORPUS_PATH)

    dictionary_jt = gensim.corpora.Dictionary.load(JT_DICT_PATH)
    corpus_jt = gensim.corpora.MmCorpus(JT_CORPUS_PATH)

    tfidf = gensim.models.TfidfModel(corpus, normalize=True)
    corpus_tfidf = tfidf[corpus]

    tfidf_jt = gensim.models.TfidfModel(corpus_jt, normalize=True)
    corpus_jt_tfidf = tfidf_jt[corpus_jt]

    # lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=30)
    # corpus_lsi = lsi[corpus_tfidf]
    #
    # optimal_clusters = run_k_mean_and_get_optimal_k(corpus_lsi, 30)

    lda_topic_3 = gensim.models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=3)
    lda_topic_6 = gensim.models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=6)
    lda_topic_8 = gensim.models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=8)
    lda_topic_10 = gensim.models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=10)

    lda_jt_3 = gensim.models.LdaModel(corpus_jt_tfidf, id2word=dictionary_jt, num_topics=3)
    lda_jt_6 = gensim.models.LdaModel(corpus_jt_tfidf, id2word=dictionary_jt, num_topics=6)
    lda_jt_8 = gensim.models.LdaModel(corpus_jt_tfidf, id2word=dictionary_jt, num_topics=8)
    lda_jt_10 = gensim.models.LdaModel(corpus_jt_tfidf, id2word=dictionary_jt, num_topics=10)

    # corpus_lda = lda[corpus_tfidf]
    # labels = run_k_mean_with_k(corpus_lda,100, 3)

    # max_l = [max(doc, key=lambda i: abs(i[1])) for doc in corpus_lda]
    # idx = 0
    # for jname in extract_all_jtpair(DATA_DIR):
    #     print "{}\t{}".format(max_l[idx][0], jname)
    #     # print "{}\t{}".format(labels[idx], jname)
    #     idx += 1

    # lda.save(LDA_PATH)
    lda_topic_3.save(os.path.join(LDA_DIR, 'topics_3.lda'))
    lda_topic_6.save(os.path.join(LDA_DIR, 'topics_6.lda'))
    lda_topic_8.save(os.path.join(LDA_DIR, 'topics_8.lda'))
    lda_topic_10.save(os.path.join(LDA_DIR, 'topics_10.lda'))

    lda_jt_3.save(os.path.join(LDA_DIR, 'jt_3.lda'))
    lda_jt_6.save(os.path.join(LDA_DIR, 'jt_6.lda'))
    lda_jt_8.save(os.path.join(LDA_DIR, 'jt_8.lda'))
    lda_jt_10.save(os.path.join(LDA_DIR, 'jt_10.lda'))





if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    main()
