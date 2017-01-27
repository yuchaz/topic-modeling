import pyLDAvis.gensim
from gensim import corpora, models


CORPUS_PATH = './storage/models/text_corpus.mm'
DICT_PATH = './storage/models/text_corpus.dict'
LDA_MODEL_PATH = './storage/models/topics_8.lda'
def main():
    corpus = corpora.MmCorpus(CORPUS_PATH)
    dictionary = corpora.Dictionary.load(DICT_PATH)
    lda = models.LdaModel.load(LDA_MODEL_PATH)
    pub_topics = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.show(pub_topics)



if __name__ == '__main__':
    main()
