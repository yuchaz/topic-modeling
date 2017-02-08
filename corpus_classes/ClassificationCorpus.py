import os
import gensim
import numpy as np
import packages.data_path_parser as dp


models_dir = dp.get_models_dir()

class ClassificationCorpus(object):
    def __init__(self,filename,model='tfidf',dim=100,if_tfidf=True):
        self.dictionary, self.corpus, self.categories = \
            get_dictionary_and_corpus(filename)
        self.model = model
        self.dim = dim
        self.transformer,self.transformed_corpus = model_transformer(
            self.corpus,model,dim,if_tfidf)

    def sparse(self,num_terms=None):
        return gensim.matutils.corpus2csc(
            self.transformed_corpus, num_terms=num_terms).transpose()

    def transform(self,corpus):
        if not self.transformer:
            return self.transformer[corpus]
        else:
            return corpus

def model_transformer(corpus,model,num_topics=100,if_tfidf=True):
    if model == 'bow':
        return None,corpus
    elif if_tfidf or model == 'tfidf':
        transformer = gensim.models.TfidfModel(corpus,normalize=True)
        transformed_corpus = transformer[corpus]
        if model == 'tfidf':
            return transformer, transformed_corpus
        corpus = transformed_corpus

    if model == 'lda':
        transformer = gensim.models.LdaModel(corpus, num_topics=num_topics)
    elif model == 'lsi':
        transformer = gensim.models.LsiModel(corpus,num_topics=num_topics)
    return transformer,transformer[corpus]

def get_dictionary_and_corpus(filename):
    dictionary = gensim.corpora.Dictionary.load(os.path.join(models_dir, filename+'.dict'))
    corpus = gensim.corpora.MmCorpus(os.path.join(models_dir, filename+'.mm'))
    with open(os.path.join(models_dir, filename+'.clf'), 'r+') as deci_file:
        categories = np.array(deci_file.read().split('\n'))
    deci_file.close()
    return dictionary, corpus, categories
