import os
import gensim

MODELS_DIR = './storage/models'

def main():
    dictionary = gensim.corpora.Dictionary.load(os.path.join(MODELS_DIR,
                                                     "mtsamples.dict"))
    corpus = gensim.corpora.MmCorpus(os.path.join(MODELS_DIR, "mtsamples.mm"))
    tfidf = gensim.models.TfidfModel(corpus, normalize=True)

    corpus_tfidf = tfidf[corpus]
    lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=3)

    corpus_lsi = lsi[corpus_tfidf]
    # import pdb; pdb.set_trace()
    max_l = [max(doc, key=lambda i: abs(i[1])) for doc in corpus_lsi]
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
