from packages.PublicationCorpus import PublicationCorpus
import packages.data_path_parser as dp
import os
import nltk
import gensim

TEXT_DIR = './storage/texts/'
JT_DIR= './storage/journalname_title/'
MODELS_DIR = './storage/models'
DATA_DIR = dp.get_training_corpus()

def main():
    stoplist = set(nltk.corpus.stopwords.words("english"))
    journal_stoplist = set(
        """letters journal annals international current opinion
           equilibria fig eq et al ev nm gev using application model
           model new research applied study effect high low analysis
           relationship background modeling one two three four five six
           seven eight nine ten hundred thousand million billion based
           design equations equation figure figures"""
        .split())
    stoplist.update(journal_stoplist)

    text_corpus = PublicationCorpus(TEXT_DIR, stoplist)
    jt_corpus = PublicationCorpus(JT_DIR, stoplist)

    text_corpus.dictionary.filter_extremes(no_below=3, no_above=0.5)
    jt_corpus.dictionary.filter_extremes(no_below=3, no_above=0.5)

    text_corpus.dictionary.save(os.path.join(MODELS_DIR, "text_corpus.dict"))
    gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "text_corpus.mm"),
                                      text_corpus)

    jt_corpus.dictionary.save(os.path.join(MODELS_DIR, "jt_corpus.dict"))
    gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "jt_corpus.mm"),
                                      jt_corpus)
    with open('./storage/models/text_corpus.classify', 'w+') as textf:
        textf.write('\n'.join(text_corpus.journal_categories))
    textf.close()
    with open('./storage/models/jt_corpus.classify', 'w+') as jtf:
        jtf.write('\n'.join(jt_corpus.journal_categories))
    jtf.close()

if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    main()
