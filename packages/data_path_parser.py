import json
import ConfigParser
import os

CONFIG_PATH = './config.ini'
DATA_SECTION = 'data_dir'
HOMEDIR_ELEMENT = 'homedir'
TRAINING_CORPUS = 'train'
DEV_CORPUS = 'dev'
TESTING_CORPUS = 'test'

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
