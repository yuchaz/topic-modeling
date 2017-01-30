import csv, json, os

filename = 'journal_list_by_subject_three_class'
data_dir = './data'
annotated_csv_path = os.path.join(data_dir, filename+'.csv')
annotated_json_path = os.path.join(data_dir, filename+'.json')

class AnnotatedJournalEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__

class AnnotatedJournal(object):
    def __init__(self, csv_row):
        self.journalname = csv_row[0]
        self.is_cse = csv_row[1]
        self.is_physics = csv_row[2]
        self.is_mse = csv_row[3]
        self.category = csv_row[4]
    def to_json():
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)

    def calc_category_score():
        return 2**2*self.is_cse+2*self.is_physics+is_mse

def parse_csv():
    with open(annotated_csv_path, 'r') as csvfile:
        annotated_data = csv.reader(csvfile, delimiter=',')
        journal_annotated_with_subject = {row[0]: AnnotatedJournal(row) for row in annotated_data}
    csvfile.close()
    with open(annotated_json_path, 'w+') as jsonfile:
        jsonfile.write(json.dumps(journal_annotated_with_subject, cls=AnnotatedJournalEncoder).encode('utf-8'))
    jsonfile.close()
    return journal_annotated_with_subject

if __name__ == '__main__':
    parse_csv()
