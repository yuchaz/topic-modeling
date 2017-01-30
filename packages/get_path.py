import packages.data_path_parser as dp
import os

STORAGE_DIR = './storage'
PREPROCESSED_TEXT_DIR = os.path.join(STORAGE_DIR, 'text')
PREPROCESSED_JT_DIR = os.path.join(STORAGE_DIR, 'journalname_title')
MODELS_DIR = os.path.join(STORAGE_DIR, 'models')

TRAINING_SUBDIR = 'train'
DEV_SUBDIR = 'dev'
TESTING_SUBDIR = 'test'

CorpusPathHandler = {
    "training_corpus": dp.get_training_corpus(),
    "dev_corpus": dp.get_dev_corpus(),
    "test_corpus": dp.get_test_corpus(),

    "training_text_corpus": os.path.join(PREPROCESSED_TEXT_DIR, TRAINING_SUBDIR),
    "dev_text_corpus": os.path.join(PREPROCESSED_TEXT_DIR, DEV_SUBDIR),
    "test_text_corpus": os.path.join(PREPROCESSED_TEXT_DIR, TESTING_SUBDIR),

    "training_jt_corpus": os.path.join(PREPROCESSED_JT_DIR, TRAINING_SUBDIR),
    "dev_jt_corpus": os.path.join(PREPROCESSED_JT_DIR, DEV_SUBDIR),
    "test_jt_corpus": os.path.join(PREPROCESSED_JT_DIR, TESTING_SUBDIR),

}

def annotated_training_set_location():
    return CorpusPathHandler["training_text_corpus"]

def annotated_dev_set_location():
    return CorpusPathHandler["dev_text_corpus"]

def annotated_test_set_location():
    return CorpusPathHandler["test_text_corpus"]
