from packages.extract_texts import extract_all_jtpair
from packages.k_means import run_k_mean_and_get_optimal_k, run_k_mean_with_k
import packages.data_path_parser as dp
import gensim
import os
from scipy.sparse import csr_matrix
from sklearn.svm import SVC
import numpy as np

TRAIN_TEXT_DICT_PATH = './storage/models/text_corpus_train.dict'
TRAIN_TEXT_CORPUS_PATH = './storage/models/text_corpus_train.mm'
TEST_TEXT_DICT_PATH = './storage/models/text_corpus_test.dict'
TEST_TEXT_CORPUS_PATH = './storage/models/text_corpus_test.mm'

# JT_DICT_PATH = './storage/models/jt_corpus.dict'
# JT_CORPUS_PATH = './storage/models/jt_corpus.mm'
LDA_DIR = './storage/models/'


def main():
    train_dictionary = gensim.corpora.Dictionary.load(TRAIN_TEXT_DICT_PATH)
    train_corpus = gensim.corpora.MmCorpus(TRAIN_TEXT_CORPUS_PATH)
    with open(os.path.join(LDA_DIR, 'text_corpus_train.classify'), 'r+') as decif:
        train_categories = np.array(decif.read().split('\n'))
    decif.close()

    test_dictionary = gensim.corpora.Dictionary.load(TEST_TEXT_DICT_PATH)
    test_corpus = gensim.corpora.MmCorpus(TEST_TEXT_CORPUS_PATH)
    with open(os.path.join(LDA_DIR, 'text_corpus_test.classify'), 'r+') as decif2:
        test_categories = np.array(decif2.read().split('\n'))
    decif2.close()

    # dictionary_jt = gensim.corpora.Dictionary.load(JT_DICT_PATH)
    # corpus_jt = gensim.corpora.MmCorpus(JT_CORPUS_PATH)

    # tfidf = gensim.models.TfidfModel(corpus, normalize=True)
    # corpus_tfidf = tfidf[corpus]
    train_corpus_sparse = gensim.matutils.corpus2csc(train_corpus).transpose()
    topic_classifier = SVC()
    topic_classifier.fit(train_corpus_sparse, train_categories)
    # import pdb; pdb.set_trace()
    test_corpus_sparse = gensim.matutils.corpus2csc(corpus=test_corpus,num_terms=train_corpus_sparse.shape[1]).transpose()
    predicted_categories = topic_classifier.predict(test_corpus_sparse)
    print sum(1 for i in range(len(predicted_categories)) if predicted_categories[i]==test_categories[i])

    # tfidf_jt = gensim.models.TfidfModel(corpus_jt, normalize=True)
    # corpus_jt_tfidf = tfidf_jt[corpus_jt]

    # lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=30)
    # corpus_lsi = lsi[corpus_tfidf]
    #
    # optimal_clusters = run_k_mean_and_get_optimal_k(corpus_lsi, 30)

    # lda_topic_3 = gensim.models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=3)
    # lda_topic_6 = gensim.models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=6)
    # lda_topic_8 = gensim.models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=8)
    # lda_topic_10 = gensim.models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=10)
    #
    # lda_jt_3 = gensim.models.LdaModel(corpus_jt_tfidf, id2word=dictionary_jt, num_topics=3)
    # lda_jt_6 = gensim.models.LdaModel(corpus_jt_tfidf, id2word=dictionary_jt, num_topics=6)
    # lda_jt_8 = gensim.models.LdaModel(corpus_jt_tfidf, id2word=dictionary_jt, num_topics=8)
    # lda_jt_10 = gensim.models.LdaModel(corpus_jt_tfidf, id2word=dictionary_jt, num_topics=10)
    #
    # corpus_lda = lda[corpus_tfidf]
    # labels = run_k_mean_with_k(corpus_lda,100, 3)

    # max_l = [max(doc, key=lambda i: abs(i[1])) for doc in corpus_lda]
    # idx = 0
    # for jname in extract_all_jtpair(DATA_DIR):
    #     print "{}\t{}".format(max_l[idx][0], jname)
    #     # print "{}\t{}".format(labels[idx], jname)
    #     idx += 1

    # lda.save(LDA_PATH)
    # lda_topic_3.save(os.path.join(LDA_DIR, 'topics_3.lda'))
    # lda_topic_6.save(os.path.join(LDA_DIR, 'topics_6.lda'))
    # lda_topic_8.save(os.path.join(LDA_DIR, 'topics_8.lda'))
    # lda_topic_10.save(os.path.join(LDA_DIR, 'topics_10.lda'))
    #
    # lda_jt_3.save(os.path.join(LDA_DIR, 'jt_3.lda'))
    # lda_jt_6.save(os.path.join(LDA_DIR, 'jt_6.lda'))
    # lda_jt_8.save(os.path.join(LDA_DIR, 'jt_8.lda'))
    # lda_jt_10.save(os.path.join(LDA_DIR, 'jt_10.lda'))





if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    main()
