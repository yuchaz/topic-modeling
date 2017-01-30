from packages.PublicationCorpus import PublicationCorpus, TestCorpus
import packages.data_path_parser as dp
import os
import nltk
import gensim

TEXT_DIR_TRAIN = './storage/texts/train'
TEXT_DIR_TEST = './storage/texts/test'
JT_DIR_TRAIN = './storage/journalname_title/train'
JT_DIR_TEST = './storage/journalname_title/test'
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

    text_corpus_train = PublicationCorpus(TEXT_DIR_TRAIN, stoplist)
    jt_corpus_train = PublicationCorpus(JT_DIR_TRAIN, stoplist)

    text_corpus_train.dictionary.filter_extremes(no_below=3, no_above=0.5)
    jt_corpus_train.dictionary.filter_extremes(no_below=3, no_above=0.5)

    text_corpus_train.dictionary.save(os.path.join(MODELS_DIR, "text_corpus_train.dict"))
    gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "text_corpus_train.mm"),
                                      text_corpus_train)

    jt_corpus_train.dictionary.save(os.path.join(MODELS_DIR, "jt_corpus_train.dict"))
    gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "jt_corpus_train.mm"),
                                      jt_corpus_train)
    with open('./storage/models/text_corpus_train.classify', 'w+') as textf:
        textf.write('\n'.join(text_corpus_train.journal_categories))
    textf.close()
    with open('./storage/models/jt_corpus_train.classify', 'w+') as jtf:
        jtf.write('\n'.join(jt_corpus_train.journal_categories))
    jtf.close()

    text_corpus_test = TestCorpus(TEXT_DIR_TEST, stoplist, text_corpus_train.dictionary)
    jt_corpus_test = TestCorpus(JT_DIR_TEST, stoplist, jt_corpus_train.dictionary)

    text_corpus_test.dictionary.filter_extremes(no_below=3, no_above=0.5)
    jt_corpus_test.dictionary.filter_extremes(no_below=3, no_above=0.5)

    text_corpus_test.dictionary.save(os.path.join(MODELS_DIR, "text_corpus_test.dict"))
    gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "text_corpus_test.mm"),
                                      text_corpus_test)

    jt_corpus_test.dictionary.save(os.path.join(MODELS_DIR, "jt_corpus_test.dict"))
    gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "jt_corpus_test.mm"),
                                      jt_corpus_test)
    with open('./storage/models/text_corpus_test.classify', 'w+') as textf2:
        textf2.write('\n'.join(text_corpus_test.journal_categories))
    textf2.close()
    with open('./storage/models/jt_corpus_test.classify', 'w+') as jtf2:
        jtf2.write('\n'.join(jt_corpus_test.journal_categories))
    jtf2.close()

if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    main()
