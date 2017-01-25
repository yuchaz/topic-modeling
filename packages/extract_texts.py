from scienceie2017_scripts.util import parseXML
import os

TEXTS_DIR = './storage/texts/'
DATA_DIR = './scienceie2017_data/dev/'

def extract_text_from_xml(fpath):
    Handler = parseXML(fpath)
    return (" ".join([t for n, t in Handler.text.items()])).encode('utf-8')

def extract_all_texts(dirpath=DATA_DIR):
    file_paths = os.listdir(dirpath)
    yield (extractTextFromXml(fpath) for fpath in file_paths)

def save_texts(datapath=DATA_DIR, textpath=TEXTS_DIR):
    for dpath in os.listdir(datapath):
        if not dpath.endswith(".xml"):
            continue
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
