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

    tfidf_train = gensim.models.TfidfModel(train_corpus, normalize=True)
    train_corpus_tfidf = tfidf_train[train_corpus]

    tfidf_test = gensim.models.TfidfModel(test_corpus, normalize=True)
    test_corpus_tfidf = tfidf_test[test_corpus]

    train_corpus_sparse = gensim.matutils.corpus2csc(train_corpus_tfidf).transpose()
    topic_classifier = SVC()
    topic_classifier.fit(train_corpus_sparse, train_categories)

    test_corpus_sparse = gensim.matutils.corpus2csc(corpus=test_corpus_tfidf,num_terms=train_corpus_sparse.shape[1]).transpose()
    predicted_categories = topic_classifier.predict(train_corpus_sparse)
    print sum(1 for i in range(len(predicted_categories)) if predicted_categories[i]==train_categories[i])

if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    main()
