import json
import ConfigParser
import os

CONFIG_PATH = './config.ini'
DATA_SECTION = 'data_dir'
HOMEDIR_ELEMENT = 'homedir'
TRAINING_CORPUS = 'train'
DEV_CORPUS = 'dev'
TESTING_CORPUS = 'test'

STORAGE_DIR = './storage'
PREPROCESSED_TEXT_DIR = os.path.join(STORAGE_DIR, 'texts')
PREPROCESSED_JT_DIR = os.path.join(STORAGE_DIR, 'journalname_title')
MODELS_DIR = os.path.join(STORAGE_DIR, 'models')

def read_config(path):
    config = ConfigParser.SafeConfigParser()
    config.read(path)
    return config

def read_home_dir():
    config = read_config(CONFIG_PATH)
    return config.get(DATA_SECTION, HOMEDIR_ELEMENT)

def get_training_corpus():
    return os.path.join(read_home_dir(), TRAINING_CORPUS)

def get_dev_corpus():
    return os.path.join(read_home_dir(), DEV_CORPUS)

def get_test_corpus():
    return os.path.join(read_home_dir(), TESTING_CORPUS)

CorpusPathHandler = {
    "training_text_corpus": os.path.join(PREPROCESSED_TEXT_DIR, TRAINING_CORPUS),
    "dev_text_corpus": os.path.join(PREPROCESSED_TEXT_DIR, DEV_CORPUS),
    "test_text_corpus": os.path.join(PREPROCESSED_TEXT_DIR, TESTING_CORPUS),

    "training_jt_corpus": os.path.join(PREPROCESSED_JT_DIR, TRAINING_CORPUS),
    "dev_jt_corpus": os.path.join(PREPROCESSED_JT_DIR, DEV_CORPUS),
    "test_jt_corpus": os.path.join(PREPROCESSED_JT_DIR, TESTING_CORPUS),
}

def annotated_text_training():
    return CorpusPathHandler["training_text_corpus"]

def annotated_text_dev():
    return CorpusPathHandler["dev_text_corpus"]

def annotated_text_test():
    return CorpusPathHandler["test_text_corpus"]

def annotated_jt_training():
    return CorpusPathHandler["training_jt_corpus"]

def annotated_jt_dev():
    return CorpusPathHandler["dev_jt_corpus"]

def annotated_jt_test():
    return CorpusPathHandler["test_jt_corpus"]

def get_annotated_training_set():
    return annotated_text_training()

def get_annotated_dev_set():
    return annotated_text_dev()

def get_annotated_test_set():
    return annotated_text_test()

def get_models_dir():
    return MODELS_DIR
