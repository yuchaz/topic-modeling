from scienceie2017_scripts.util import parseXML
import os
import gensim

TEXTS_DIR = './storage/texts/'
DATA_DIR = './scienceie2017_data/dev/'
JOURNALNAME_TITLE_DIR = './storage/journalname_title/'

def extract_journalname_title_pair_from_xml(fpath):
    Handler = parseXML(fpath)
    return (Handler.journalname + " " + Handler.title).encode('utf-8')

def extract_journal_title_from_xml(fpath):
    Handler = parseXML(fpath)
    return Handler.journalname

def extract_paper_title_from_xml(fpath):
    Handler = parseXML(fpath)
    return Handler.title

def extract_text_from_xml(fpath):
    Handler = parseXML(fpath)
    return (" ".join([t for n, t in Handler.text.items()])).encode('utf-8')

def extract_all_texts(dirpath=DATA_DIR):
    file_paths = os.listdir(dirpath)
    yield (extractTextFromXml(fpath) for fpath in file_paths)

def save_title_and_journal_name(data_dir=DATA_DIR, jt_dir=JOURNALNAME_TITLE_DIR):
    for data_filename in os.listdir(data_dir):
        if not data_filename.endswith(".xml"): continue

        jt_filename = os.path.splitext(data_filename)[0]+".txt"
        jt_path = os.path.join(jt_dir, jt_filename)
        if not os.path.exists(os.path.dirname(jt_path)):
            try:
                os.makedirs(os.path.dirname(jt_path))
            except OSError as exc:
                if exc.errno != errno.EEXIST: raise
        with open (jt_path, 'w+') as jt_file:
            jt_file.write(extract_journalname_title_pair_from_xml(os.path.join(data_dir, data_filename)))
        jt_file.close()

def save_texts(datapath=DATA_DIR, textpath=TEXTS_DIR):
    for dpath in os.listdir(datapath):
        if not dpath.endswith(".xml"): continue
        tpath = os.path.splitext(dpath)[0] + ".txt"
        output_file_name = os.path.join(TEXTS_DIR, tpath)
        if not os.path.exists(os.path.dirname(output_file_name)):
            try:
                os.makedirs(os.path.dirname(output_file_name))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(output_file_name, 'w+') as tfile:
            tfile.write(extract_text_from_xml(os.path.join(DATA_DIR, dpath)))
        tfile.close()
