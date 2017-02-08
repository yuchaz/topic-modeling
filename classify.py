from sklearn import svm, model_selection, naive_bayes, metrics
from corpus_classes.ClassificationCorpus import ClassificationCorpus

training_corpus_name = 'training_corpus'
evaluation_corpus_name = 'evaluation_corpus'
reduced_dimension = 100
model_to_use = 'bow'
if_tfidf = True

def main():
    train_corpus = ClassificationCorpus(training_corpus_name,model=model_to_use)
    evaluation_corpus = ClassificationCorpus(evaluation_corpus_name,model=model_to_use)

    training_corpus_sparse = train_corpus.sparse()

    labeling_corpus(training_corpus_sparse, train_corpus.categories,
                    evaluation_corpus.sparse(num_terms=training_corpus_sparse.shape[1]),
                    evaluation_corpus.categories)


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
