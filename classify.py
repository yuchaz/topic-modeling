from sklearn import svm, model_selection, naive_bayes, metrics
from corpus_classes.ClassificationCorpus import ClassificationCorpus

training_corpus_name = 'training_corpus'
evaluation_corpus_name = 'evaluation_corpus'
reduced_dimension = 1000
model_to_use = 'bow'
if_tfidf = False

param_grid_svm = {'C':[1e-1,1,10,1e3,1e4,1e5], 'gamma':[0,1e-3,1e-2,1e-1,1,10,100,'auto']}
param_grid_nb = {'alpha':[1e-9,1e-8,1e-7,1e-6,5e-6,1e-5,5e-5,1e-4]}

def main():
    train_corpus = ClassificationCorpus(training_corpus_name,model=model_to_use)
    evaluation_corpus = ClassificationCorpus(evaluation_corpus_name,model=model_to_use)

    training_corpus_sparse = train_corpus.sparse()

    labeling_corpus(training_corpus_sparse, train_corpus.categories,
                    evaluation_corpus.sparse(num_terms=training_corpus_sparse.shape[1]),
                    evaluation_corpus.categories)


def labeling_corpus(training_corpus, training_categories, evaluation_corpus, evaluation_categories):
    estimator, param_grid = naive_bayes_estimator()
    # estimator, param_grid = svm_estimator()

    topic_classifier = estimator
    topic_classifier = model_selection.GridSearchCV(estimator=estimator, param_grid=param_grid,n_jobs=-1, cv=10)
    topic_classifier.fit(training_corpus, training_categories)
    pred_categories = topic_classifier.predict(evaluation_corpus)
    confusion_matrix = metrics.confusion_matrix(evaluation_categories, pred_categories)
    print topic_classifier.score(evaluation_corpus, evaluation_categories)
    print topic_classifier.score(training_corpus, training_categories)
    print pred_categories
    if getattr(topic_classifier,'best_params_',None): print topic_classifier.best_params_
    if getattr(topic_classifier,'best_score_',None): print topic_classifier.best_score_
    print confusion_matrix

def naive_bayes_estimator():
    return naive_bayes.MultinomialNB(alpha=1e-7), param_grid_nb
    # return naive_bayes.MultinomialNB(alpha=0.005), param_grid_nb

def svm_estimator():
    return svm.SVC(decision_function_shape='ovo', C=10, gamma='auto'), param_grid_svm

if __name__ == '__main__':
    try:
        import sys
        reload(sys)
        sys.setdefaultencoding('utf-8')
        main()
    except:
        import sys, pdb, traceback
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
