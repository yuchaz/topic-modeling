from scienceie2017_scripts.util import parseXML
import os, shutil
import gensim
import packages.data_path_parser as dp
from packages.csv_parser import parse_csv

DATA_DIR = dp.get_training_corpus()
TEXTS_DIR = './storage/texts/train/'
JOURNALNAME_TITLE_DIR = './storage/journalname_title/train/'

map_journalname_to_category = parse_csv()

def extract_journalname_from_xml(fpath):
    Handler = parseXML(fpath)
    return Handler.journalname

def extract_all_journalname(dirpath=DATA_DIR):
    file_paths = os.listdir(dirpath)
    for fn in file_paths:
        if not fn.endswith('.xml'): continue
        yield extract_journalname_from_xml(os.path.join(dirpath, fn))

def extract_journalname_title_pair_from_xml(fpath):
    Handler = parseXML(fpath)
    return (Handler.journalname + "\t" + Handler.title).encode('utf-8')

def extract_all_jtpair(dirpath=DATA_DIR):
    file_paths = os.listdir(dirpath)
    for fn in file_paths:
        if not fn.endswith(".xml"): continue
        yield extract_journalname_title_pair_from_xml(os.path.join(dirpath, fn))

def extract_journal_title_from_xml(fpath):
    Handler = parseXML(fpath)
    return Handler.journalname

def extract_full_journal_name(dirpath=DATA_DIR):
    file_paths = os.listdir(dirpath)
    for fn in file_paths:
        if not fn.endswith(".xml"): continue
        yield extract_journal_title_from_xml(os.path.join(dirpath, fn))

def extract_paper_title_from_xml(fpath):
    Handler = parseXML(fpath)
    return (Handler.title).encode('utf-8')

def extract_text_from_xml(fpath):
    Handler = parseXML(fpath)
    return (" ".join([t for n, t in Handler.text.items()])).encode('utf-8')

def extract_all_texts(dirpath=DATA_DIR):
    file_paths = os.listdir(dirpath)
    yield (extract_text_from_xml(fpath) for fpath in file_paths)

def save_title_and_journal_name(data_dir=DATA_DIR, jt_dir=JOURNALNAME_TITLE_DIR):
    kill_files_in_output_before_write(jt_dir)
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
            # jt_file.write(extract_journalname_title_pair_from_xml(os.path.join(data_dir, data_filename)))
            to_write = u'{}\t\t\t{}'.format(extract_paper_title_from_xml(os.path.join(data_dir, data_filename)),
                map_journalname_to_category.get(
                    extract_journalname_from_xml(os.path.join(data_dir, data_filename))).category)
            jt_file.write(to_write)
        jt_file.close()

def save_texts(datapath=DATA_DIR, textpath=TEXTS_DIR):
    kill_files_in_output_before_write(textpath)
    for dpath in os.listdir(datapath):
        if not dpath.endswith(".xml"): continue
        tpath = os.path.splitext(dpath)[0] + ".txt"
        output_file_name = os.path.join(textpath, tpath)
        if not os.path.exists(os.path.dirname(output_file_name)):
            try:
                os.makedirs(os.path.dirname(output_file_name))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(output_file_name, 'w+') as tfile:
            to_write = u'{}\t\t\t{}'.format(extract_text_from_xml(os.path.join(datapath, dpath)),
                map_journalname_to_category.get(
                    extract_journalname_from_xml(os.path.join(datapath, dpath))).category)
            tfile.write(to_write)
        tfile.close()

def kill_files_in_output_before_write(path):
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
