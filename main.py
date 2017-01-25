from packages.PublicationCorpus import PublicationCorpus
import os
import nltk
import gensim
import pprint

# HOMEDIR = './storage/texts/'
HOMEDIR = './storage/journalname_title/'
MODELS_DIR = './storage/models'
def main():
    stoplist = set(nltk.corpus.stopwords.words("english"))
    corpus = PublicationCorpus(HOMEDIR, stoplist)
    corpus.dictionary.save(os.path.join(MODELS_DIR, "pubcorpus.dict"))
    gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "pubcorpus.mm"),
                                      corpus)

    dictionary = corpus.dictionary
    tfidf = gensim.models.TfidfModel(corpus, normalize=True)
    # tfidf is a transformation from bow to tfidf.
    corpus_tfidf = tfidf[corpus]
    lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=3)
    # transformation from tfidf to lsi
    corpus_lsi = lsi[corpus_tfidf]
    max_l = [max(doc, key=lambda i: abs(i[1])) for doc in corpus_lsi]
    print max_l
    print lsi.print_topics()

if __name__ == '__main__':
    main()
