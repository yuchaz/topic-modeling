from packages.extract_texts import save_texts, save_title_and_journal_name
import packages.data_path_parser as dp

train_raw_data_path = dp.get_test_corpus()
dev_raw_data_path = dp.get_dev_corpus()
test_raw_data_path = dp.get_test_corpus()

path_annotated_texts_for_training = dp.annotated_text_training()
path_annotated_texts_for_dev = dp.annotated_text_dev()
path_annotated_texts_for_test = dp.annotated_text_test()

path_annotated_jt_for_training = dp.annotated_jt_training()
path_annotated_jt_for_dev = dp.annotated_jt_dev()
path_annotated_jt_for_test = dp.annotated_jt_test()

def main():
    save_texts(train_raw_data_path, path_annotated_texts_for_training)
    save_title_and_journal_name(train_raw_data_path, path_annotated_jt_for_training)
    save_texts(dev_raw_data_path, path_annotated_texts_for_dev)
    save_title_and_journal_name(dev_raw_data_path, path_annotated_jt_for_dev)
    save_texts(test_raw_data_path, path_annotated_texts_for_test)
    save_title_and_journal_name(test_raw_data_path, path_annotated_jt_for_test)

if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    main()
