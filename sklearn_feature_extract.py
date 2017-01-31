from sklearn.feature_extraction.text import CountVectorizer
from packages.PublicationCorpus import SklearnCorpus
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import packages.data_path_parser as dp
from sklearn.metrics import confusion_matrix

annotated_data_for_training = dp.get_annotated_training_set()
annotated_data_for_evaluation = dp.get_annotated_dev_set()
models_dir = dp.get_models_dir()


def main():
    vectorizer = CountVectorizer(min_df=1)
    train_corpus = SklearnCorpus(annotated_data_for_training)
    eval_corpus = SklearnCorpus(annotated_data_for_evaluation)
    X = vectorizer.fit_transform(train_corpus)
    X_eval = vectorizer.transform(eval_corpus)
    Y = train_corpus.journal_categories
    y_eval = eval_corpus.journal_categories
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    # X_train = X
    # y_train = Y
    svm = train_svm(X_train, y_train)
    pred = svm.predict(X_eval)
    print pred
    print(svm.score(X_eval, y_eval))
    print(confusion_matrix(pred, y_eval))
    # print(svm.best_score_)
    # print(svm.best_params_)


def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    param_grid = {'gamma':[0,1e-3,1e-2,1e-1,1,10,100,'auto']}
    svm = SVC(C=1000000.0, kernel='rbf', gamma=1e-2)
    svm.fit(X, y)
    topic_classifier = GridSearchCV(estimator=svm, param_grid=param_grid,n_jobs=-1, cv=10)
    topic_classifier.fit(X,y)
    # return svm
    return topic_classifier



if __name__ == '__main__':
    main()
