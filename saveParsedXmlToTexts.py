from packages.extract_texts import save_texts, save_title_and_journal_name
import packages.data_path_parser as dp

test_dir = dp.get_dev_corpus()
test_texts_dir = './storage/texts/test/'
test_jt_dir = './storage/journalname_title/test/'

def main():
    save_texts()
    save_title_and_journal_name()
    save_texts(datapath=test_dir, textpath=test_texts_dir)
    save_title_and_journal_name(data_dir=test_dir, jt_dir=test_jt_dir)

if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    main()
