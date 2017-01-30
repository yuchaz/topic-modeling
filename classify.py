from packages.extract_texts import extract_all_jtpair
from packages.k_means import run_k_mean_and_get_optimal_k, run_k_mean_with_k
import packages.data_path_parser as dp
import gensim
import os
from scipy.sparse import csr_matrix
from sklearn import svm
import numpy as np

models_dir = dp.get_models_dir()
TRAIN_TEXT_DICT_PATH = './storage/models/training_corpus.dict'
TRAIN_TEXT_CORPUS_PATH = './storage/models/training_corpus.mm'
TEST_TEXT_CORPUS_PATH = './storage/models/evaluation_corpus.mm'
training_corpus_categories = os.path.join(models_dir, 'training_corpus.clf')
evaluation_corpus_categories = os.path.join(models_dir, 'evaluation_corpus.clf')

training_corpus_name = 'training_corpus'
evaluation_corpus_name = 'evaluation_corpus'

def main():
    train_dictionary, training_corpus, training_categories = get_dictionary_and_corpus(training_corpus_name)
    ed, evaluation_corpus, evaluation_categories = get_dictionary_and_corpus(evaluation_corpus_name)

    tfidf_train, training_corpus_tfidf = calc_tfidf_matrix(training_corpus)
    tfidf_eval, evaluation_corpus_tfidf = calc_tfidf_matrix(evaluation_corpus)

    # training_corpus_sparse = gensim.matutils.corpus2csc(training_corpus).transpose()
    # evaluation_corpus_sparse = gensim.matutils.corpus2csc(corpus=evaluation_corpus,num_terms=training_corpus_sparse.shape[1]).transpose()
    training_corpus_sparse = gensim.matutils.corpus2csc(training_corpus_tfidf).transpose()
    evaluation_corpus_sparse = gensim.matutils.corpus2csc(corpus=evaluation_corpus_tfidf,num_terms=training_corpus_sparse.shape[1]).transpose()

    eval_pair = evaluation_corpus_sparse, evaluation_categories
    train_pair = training_corpus_sparse, training_categories

    score = labeling_corpus(training_corpus_sparse, training_categories, *eval_pair)
    print score

def get_dictionary_and_corpus(filename):
    dictionary = gensim.corpora.Dictionary.load(os.path.join(models_dir, filename+'.dict'))
    corpus = gensim.corpora.MmCorpus(os.path.join(models_dir, filename+'.mm'))
    with open(os.path.join(models_dir, filename+'.clf'), 'r+') as deci_file:
        categories = np.array(deci_file.read().split('\n'))
    deci_file.close()
    return dictionary, corpus, categories

def labeling_corpus(training_corpus, training_categories, evaluation_corpus, evaluation_categories):
    topic_classifier = svm.SVC(decision_function_shape='ovo')
    topic_classifier.fit(training_corpus, training_categories)
    predicted_categories = topic_classifier.predict(evaluation_corpus)

    return float(sum(1 for i in range(len(predicted_categories)) if predicted_categories[i]==evaluation_categories[i]))/len(predicted_categories)

def calc_tfidf_matrix(bow_corpus):
    tfidf = gensim.models.TfidfModel(bow_corpus, normalize=True)
    return tfidf, tfidf[bow_corpus]
    
if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    main()
