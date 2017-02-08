from packages.PublicationCorpus import PublicationCorpus, TestCorpus
import packages.data_path_parser as dp
import os
import nltk
import gensim
from packages.extract_texts import kill_files_in_output_before_write

annotated_data_for_training = [dp.get_annotated_training_set()]
annotated_data_for_evaluation = dp.get_annotated_dev_set()
models_dir = dp.get_models_dir()

def main():
    kill_files_in_output_before_write(models_dir)
    stoplist = set(nltk.corpus.stopwords.words("english"))

    training_corpus = PublicationCorpus(stoplist,*annotated_data_for_training)
    training_corpus.dictionary.save(os.path.join(models_dir, "training_corpus.dict"))
    gensim.corpora.MmCorpus.serialize(os.path.join(models_dir, "training_corpus.mm"),
                                      training_corpus)

    with open(os.path.join(models_dir, "training_corpus.clf"), 'w+') as annfile_train:
        annfile_train.write('\n'.join(training_corpus.journal_categories))
    annfile_train.close()

    evaluation_corpus = TestCorpus(annotated_data_for_evaluation, stoplist, training_corpus.dictionary)

    evaluation_corpus.dictionary.save(os.path.join(models_dir, "evaluation_corpus.dict"))
    gensim.corpora.MmCorpus.serialize(os.path.join(models_dir, "evaluation_corpus.mm"),
                                      evaluation_corpus)

    with open(os.path.join(models_dir, 'evaluation_corpus.clf'), 'w+') as annfile_eval:
        annfile_eval.write('\n'.join(evaluation_corpus.journal_categories))
    annfile_eval.close()

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
