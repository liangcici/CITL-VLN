import os
import ast
import json
import argparse
import nlpaug.augmenter.word as naw
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='generate positive instructions')
    parser.add_argument('--source', type=str, help='input file')
    parser.add_argument('--target', type=str, help='output file')
    parser.add_argument('--dest_dir', type=str, help='output file')
    parser.add_argument('--aug_word_max', type=int, default=5, help='max number of augmented words')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    with open(args.source, 'r') as f:
        data = json.load(f)

    interval = 200
    aug_type_list = ['synonym', 'insert', 'substitute', 'back_translation']

    for aug_type in aug_type_list:
        if aug_type == 'synonym':
            aug = naw.SynonymAug(aug_max=args.aug_word_max)
            aug_num = 4
        elif aug_type == 'insert':
            aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert", device='cuda',
                                            aug_max=args.aug_word_max)
            aug_num = 4
        elif aug_type == 'substitute':
            aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute", device='cuda',
                                            aug_max=args.aug_word_max)
            aug_num = 4
        elif aug_type == 'back_translation':
            aug = naw.BackTranslationAug(device='cuda')
            aug_num = 1

        if not os.path.exists(os.path.join(args.dest_dir, 'temp', aug_type)):
            os.makedirs(os.path.join(args.dest_dir, 'temp', aug_type))

        for iind in range(aug_num):
            new_target = args.target.split('/')[-1].split('.')[0] + '_' + aug_type + '_' + str(iind) + '.json'

            old_instructions = []
            for data_i in tqdm(data):

                if isinstance(data_i['new_instructions'], str):
                    instrs = ast.literal_eval(data_i['new_instructions'])
                    for instr_i in instrs:
                        for instr_ii in instr_i:
                            instr_ii_old = ' '.join(instr_ii)
                            old_instructions.append(instr_ii_old)
                else:
                    instrs = data_i['new_instructions']
                    for instr_i in instrs:
                        for instr_ii in instr_i:
                            old_instructions.append(instr_ii)

            print('Load original instructions done!')
            new_instructions = []
            j_index = 0
            while j_index < len(old_instructions):
                end = j_index + interval
                if end > len(old_instructions):
                    end = len(old_instructions)
                new_instructions += aug.augment(old_instructions[j_index:end])
                j_index = end
                print('{} instructions done!'.format(j_index))
            print('Transformation done!')
            assert len(new_instructions) == len(old_instructions)

            with open(os.path.join(args.dest_dir, 'temp', aug_type, new_target), 'w') as f:
                json.dump(new_instructions, f, indent=4)

    # save to one file
    instr_list = []
    first_file = True
    for aug_t in aug_type_list:
        aug_dir = os.path.join(args.dest_dir, 'temp', aug_t)
        for aug_file in os.listdir(aug_dir):
            with open(os.path.join(aug_dir, aug_file), 'r') as f:
                data = json.load(f)

            for ind, data_i in enumerate(data):
                if first_file:
                    instr_list.append([data_i])
                else:
                    instr_list[ind].append(data_i)
            first_file = False

    print('instr_list len: {}'.format(len(instr_list)))
    print('instr_list[0] len: {}'.format(len(instr_list[0])))

    with open(os.path.join(args.dest_dir, args.target), 'w') as f:
        json.dump(instr_list, f, indent=4)

    os.system('rm -r {}'.format(os.path.join(args.dest_dir, 'temp')))
