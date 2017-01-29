from packages.extract_texts import extract_all_journalname
import packages.data_path_parser as dp
import itertools

TRAIN_DATA = dp.get_training_corpus()
DEV_DATA = dp.get_dev_corpus()
TEST_DATA = dp.get_test_corpus()

def main():
    all_data = itertools.chain(extract_all_journalname(TRAIN_DATA),
                               extract_all_journalname(DEV_DATA),
                               extract_all_journalname(TEST_DATA))

    jname_set = set([jname for jname in all_data])
    with open('journal_name_list.txt', 'w+') as jfn:
        jfn.write('\n'.join(jname_set))
    jfn.close()

if __name__ == '__main__':
    main()
