from packages.extract_texts import extract_all_jtpair
from packages.k_means import run_k_mean_and_get_optimal_k, run_k_mean_with_k
import packages.data_path_parser as dp
import gensim
import os
from scipy.sparse import csr_matrix
from sklearn import svm, model_selection, naive_bayes, metrics
import numpy as np

models_dir = dp.get_models_dir()
training_corpus_name = 'training_corpus'
evaluation_corpus_name = 'evaluation_corpus'
reduced_dimension = 100

def main():
    train_dictionary, training_corpus, training_categories = get_dictionary_and_corpus(training_corpus_name)
    ed, evaluation_corpus, evaluation_categories = get_dictionary_and_corpus(evaluation_corpus_name)

    tfidf_train, training_corpus_tfidf = calc_tfidf_matrix(training_corpus)
    tfidf_eval, evaluation_corpus_tfidf = calc_tfidf_matrix(evaluation_corpus)

    # lda_train, training_corpus_lda = transform_to_LDA(training_corpus)
    # lda_eval, evaluation_corpus_lda = transform_to_LDA(evaluation_corpus)

    # lsi_train, training_corpus_lsi = LSI_transform(training_corpus)
    # lsi_eval, evaluation_corpus_lsi = LSI_transform(evaluation_corpus)

    # ===================
    training_corpus_sparse = gensim.matutils.corpus2csc(training_corpus).transpose()
    evaluation_corpus_sparse = gensim.matutils.corpus2csc(corpus=evaluation_corpus,num_terms=training_corpus_sparse.shape[1]).transpose()
    # training_corpus_sparse = gensim.matutils.corpus2csc(training_corpus_tfidf).transpose()
    # evaluation_corpus_sparse = gensim.matutils.corpus2csc(corpus=evaluation_corpus_tfidf,num_terms=training_corpus_sparse.shape[1]).transpose()
    # training_corpus_sparse = gensim.matutils.corpus2csc(training_corpus_lda).transpose()
    # evaluation_corpus_sparse = gensim.matutils.corpus2csc(corpus=evaluation_corpus_lda,num_terms=training_corpus_sparse.shape[1]).transpose()
    # training_corpus_sparse = gensim.matutils.corpus2csc(training_corpus_lsi).transpose()
    # evaluation_corpus_sparse = gensim.matutils.corpus2csc(corpus=evaluation_corpus_lsi,num_terms=training_corpus_sparse.shape[1]).transpose()
    # ===================

    labeling_corpus(training_corpus_sparse, training_categories,
                    evaluation_corpus_sparse, evaluation_categories)

def get_dictionary_and_corpus(filename):
    dictionary = gensim.corpora.Dictionary.load(os.path.join(models_dir, filename+'.dict'))
    corpus = gensim.corpora.MmCorpus(os.path.join(models_dir, filename+'.mm'))
    with open(os.path.join(models_dir, filename+'.clf'), 'r+') as deci_file:
        categories = np.array(deci_file.read().split('\n'))
    deci_file.close()
    return dictionary, corpus, categories

def labeling_corpus(training_corpus, training_categories, evaluation_corpus, evaluation_categories):
    # param_grid = {'C':[1e-1,1,10,50,100,500], 'gamma':[0,1e-3,1e-2,1e-1,1,10,100,'auto']}
    # svc = svm.SVC(decision_function_shape='ovo', C=10, gamma='auto')
    nb = naive_bayes.MultinomialNB(alpha=0.005)
    # param_grid = {'alpha':[1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100]}

    # topic_classifier = model_selection.GridSearchCV(estimator=nb, param_grid=param_grid,n_jobs=-1, cv=10)
    topic_classifier = nb
    topic_classifier.fit(training_corpus, training_categories)
    pred_categories = topic_classifier.predict(evaluation_corpus)
    confusion_matrix = metrics.confusion_matrix(evaluation_categories, pred_categories)
    print topic_classifier.score(evaluation_corpus, evaluation_categories)
    print topic_classifier.score(training_corpus, training_categories)
    print pred_categories
    # print topic_classifier.best_params_
    # print topic_classifier.best_score_
    print confusion_matrix

def calc_tfidf_matrix(bow_corpus):
    tfidf = gensim.models.TfidfModel(bow_corpus, normalize=True)
    return tfidf, tfidf[bow_corpus]

def transform_to_LDA(bow_corpus, num_topics=reduced_dimension):
    lda = gensim.models.LdaModel(bow_corpus, num_topics=num_topics)
    return lda, lda[bow_corpus]

def LSI_transform(tfidf_corpus, num_topics=reduced_dimension):
    lsi = gensim.models.LsiModel(tfidf_corpus, num_topics=num_topics)
    return lsi, lsi[tfidf_corpus]

if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    main()
