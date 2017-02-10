from sklearn import svm, model_selection, naive_bayes, metrics
from corpus_classes.ClassificationCorpus import ClassificationCorpus

training_corpus_name = 'training_corpus'
evaluation_corpus_name = 'evaluation_corpus'
reduced_dimension = 1000
model_to_use = 'tfidf'
if_tfidf = False
ALPHA = 1e-15

param_grid_svm = {'C':[1e7,1e8,1e9,1e10,1e11,1e12], 'gamma':[0,1e-3,1e-2,1e-1,1,10,100,'auto']}
param_grid_nb = {'alpha':[1e-30,1e-20,1e-19,1e-18,1e-17,1e-16,1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,5e-6,1e-5,5e-5,1e-4]}

def main():
    train_corpus = ClassificationCorpus(training_corpus_name,model=model_to_use)
    evaluation_corpus = ClassificationCorpus(evaluation_corpus_name,model=model_to_use)

    training_corpus_sparse = train_corpus.sparse()

    predicted_categories = labeling_corpus(training_corpus_sparse, train_corpus.categories,
                                evaluation_corpus.sparse(num_terms=training_corpus_sparse.shape[1]),
                                evaluation_corpus.categories)

    with open('predicted_vs_original.txt', 'a') as po_file:
        po_file.write('Model usage: {}, alpha usage: {}, vocab_size: {}\n'.format(
            model_to_use, ALPHA, len(train_corpus.dictionary.keys())
        ))
        po_file.write('predicted\tcorrect\tjournal name\n')
        for idx in range(len(predicted_categories)):
            if predicted_categories[idx] != evaluation_corpus.categories[idx]:
                po_file.write('{}\t{}\t{}\n'.format(
                    predicted_categories[idx],
                    evaluation_corpus.categories[idx],
                    evaluation_corpus.journal_names[idx]
                ))
        po_file.write('\n\n')
    po_file.close()

def labeling_corpus(training_corpus, training_categories, evaluation_corpus, evaluation_categories):
    estimator, param_grid = naive_bayes_estimator()
    # estimator, param_grid = svm_estimator()

    topic_classifier = estimator
    # topic_classifier = model_selection.GridSearchCV(estimator=estimator, param_grid=param_grid,n_jobs=-1, cv=10)
    topic_classifier.fit(training_corpus, training_categories)
    pred_categories = topic_classifier.predict(evaluation_corpus)
    confusion_matrix = metrics.confusion_matrix(evaluation_categories, pred_categories)
    print topic_classifier.score(evaluation_corpus, evaluation_categories)
    print topic_classifier.score(training_corpus, training_categories)
    print pred_categories
    if getattr(topic_classifier,'best_params_',None): print topic_classifier.best_params_
    if getattr(topic_classifier,'best_score_',None): print topic_classifier.best_score_
    print confusion_matrix
    return topic_classifier.predict(evaluation_corpus)

def naive_bayes_estimator():
    return naive_bayes.MultinomialNB(alpha=ALPHA), param_grid_nb
    # return naive_bayes.MultinomialNB(alpha=0.005), param_grid_nb

def svm_estimator():
    return svm.SVC(decision_function_shape='ovo', C=7777777, gamma='auto'), param_grid_svm

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
