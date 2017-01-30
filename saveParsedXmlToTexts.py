from packages.extract_texts import save_texts, save_title_and_journal_name

def main():
    save_texts()
    save_title_and_journal_name()

if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    main()
