import os
import json
import copy
import argparse
from tqdm import tqdm
import stanfordnlp

from utils import print_progress
from chunking_function import create_chunk


def parse_args():
    parser = argparse.ArgumentParser(description='generate sub-instructions')
    parser.add_argument('--source', type=str, help='input file')
    parser.add_argument('--target', type=str, help='output file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    with open(args.source, 'r') as f_:
        data = json.load(f_)

    nlp = stanfordnlp.Pipeline()
    new_data = []

    total_length = len(data)

    for idx, item in tqdm(enumerate(data)):
        new_instr = []
        valid = True
        for instr in item['instructions']:
            if len(instr) > 0:
                doc = nlp(instr)

                ''' break a sentence using the chunking function '''
                instr_lemma = create_chunk(doc)

                # build the new instruction list with breakdowned sentences
                new_instr.append(instr_lemma)
            else:
                valid = False
                break

        # merge into the data dictionary
        if valid:
            new_data_i = copy.deepcopy(data[idx])
            new_sub_instr_i = []
            for new_instr_i in new_instr:
                new_sub_instrs = []
                for sub_instr_list in new_instr_i:
                    sub_instr = ' '.join(sub_instr_list)
                    new_sub_instrs.append(sub_instr)
                new_sub_instr_i.append(new_sub_instrs)
            new_data_i['new_instructions'] = new_sub_instr_i
            new_data.append(new_data_i)
        else:
            print(len(new_data))
        print_progress(idx + 1, total_length, prefix='Progress:', suffix='Complete', bar_length=50)

    print('origin data len: {}'.format(len(data)))
    print('new data len: {}'.format(len(new_data)))
    with open(args.target, 'w') as file_:
        json.dump(new_data, file_, indent=4)

