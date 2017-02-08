from packages.PublicationCorpus import TrainingCorpus, EvaluationCorpus
import packages.data_path_parser as dp
from packages.extract_texts import kill_files_in_output_before_write
import os
import gensim

models_dir = dp.get_models_dir()

def save_dictionary_and_corpus(corpus,name):
    corpus.dictionary.save(os.path.join(models_dir, name+".dict"))
    gensim.corpora.MmCorpus.serialize(os.path.join(models_dir, name+".mm"), corpus)
    with open(os.path.join(models_dir, name+".clf"), 'w+') as ann_file:
        ann_file.write('\n'.join(corpus.journal_categories))
    ann_file.close()

def main():
    kill_files_in_output_before_write(models_dir)
    training_corpus = TrainingCorpus(1500,'train')
    save_dictionary_and_corpus(training_corpus,'training_corpus')
    evaluation_corpus = EvaluationCorpus(training_corpus.dictionary,'dev')
    save_dictionary_and_corpus(evaluation_corpus,'evaluation_corpus')

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
