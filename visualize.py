import pyLDAvis.gensim
from gensim import corpora, models


text_lda = {
    'corpus':'./storage/models/text_corpus.mm',
    'dict': './storage/models/text_corpus.dict',
    'lda_model': './storage/models/topics_3.lda'
}

jt_lda = {
    'corpus':'./storage/models/jt_corpus.mm',
    'dict': './storage/models/jt_corpus.dict',
    'lda_model': './storage/models/jt_3.lda'
}

def main():
    lda_model = jt_lda
    corpus = corpora.MmCorpus(lda_model['corpus'])
    dictionary = corpora.Dictionary.load(lda_model['dict'])
    lda = models.LdaModel.load(lda_model['lda_model'])
    pub_topics = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.show(pub_topics)



if __name__ == '__main__':
    main()
