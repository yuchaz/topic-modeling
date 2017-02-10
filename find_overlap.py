import os

doc_name = ['CSE','physics','MSE']
dirname = './storage'
outputdir = './output'
term_file = [os.path.join(dirname, doc+'.txt') for doc in doc_name]


def main():
    for idx1 in range(len(doc_name)):
        for idx2 in range(len(doc_name)):
            if idx2 <= idx1: continue

            with open(term_file[idx1], 'r') as tf1:
                term1 = [term.split('\t')[1] for term in tf1.read().split('\n') if len(term) != 0]
            tf1.close()
            with open(term_file[idx2], 'r') as tf2:
                term2 = [term.split('\t')[1] for term in tf2.read().split('\n') if len(term) != 0]
            tf2.close()

            overlap = [tm for tm in term1 if tm in term2]
            with open(os.path.join(outputdir,'{}_{}'.format(
                doc_name[idx1],doc_name[idx2]
            )), 'w+') as ofn:
                ofn.write('\n'.join(overlap))
            ofn.close()
if __name__ == '__main__':
    main()
