from scienceie2017_scripts.util import parseXML
import os, shutil
import gensim
import packages.data_path_parser as dp
from packages.csv_parser import parse_csv

DATA_DIR = dp.get_training_corpus()
TEXTS_DIR = dp.annotated_text_training()
JOURNALNAME_TITLE_DIR = dp.annotated_jt_training()

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

def save_title_and_journal_name(xml_inpath=DATA_DIR, annotated_outpath=JOURNALNAME_TITLE_DIR):
    parse_xml_to_annotated_data(xml_inpath, annotated_outpath, extract_paper_title_from_xml)

def save_texts(xml_inpath=DATA_DIR, annotated_outpath=TEXTS_DIR):
    parse_xml_to_annotated_data(xml_inpath, annotated_outpath, extract_text_from_xml)

def parse_xml_to_annotated_data(xml_inpath, annotated_outpath, extract_method):
    kill_files_in_output_before_write(annotated_outpath)
    for xinfile_name in os.listdir(xml_inpath):
        if not xinfile_name.endswith('.xml'): continue
        aoutfile_name = os.path.splitext(xinfile_name)[0]+".txt"
        outfile_path = os.path.join(os.path.join(annotated_outpath, aoutfile_name))
        with open(outfile_path, 'w+') as ofile:
            content_to_write = u'{}\t\t\t{}'.format(extract_method(os.path.join(xml_inpath, xinfile_name)),
                map_journalname_to_category.get(
                    extract_journalname_from_xml(os.path.join(xml_inpath, xinfile_name))).category)
            ofile.write(content_to_write)
        ofile.close()

def kill_files_in_output_before_write(path):
    for the_file in os.listdir(path):
        if the_file == '.gitkeep': continue
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
