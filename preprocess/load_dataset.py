import json
import sys
import os
from tqdm import tqdm

from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.symbols.type_symbols import TypeSymbols
from bootleg.symbols.kg_symbols import KGSymbols


def load_jsonl(file_path):
    data = []
    with open(file_path) as jsonl_file:
        for line in jsonl_file:
            val = json.loads(line)
            data.append(val)
    print(len(data))
    return data

def load_bootleg_resources():
    input_dir = "" # fill in Bootleg model directory
    emb_dir = "" # fill in directory for Bootleg embeddding data (embs)
    entity_dump = EntitySymbols(load_dir=os.path.join(input_dir, "entity_db/entity_mappings"))
    types_wd = TypeSymbols(entity_dump, emb_dir, max_types=50, type_vocab_file="wikidatatitle_to_typeid.json",
                           type_file="wikidata_types.json")
    types_rel = TypeSymbols(entity_dump, emb_dir, max_types=50, type_vocab_file="relation_to_typeid.json",
                            type_file="kg_relation_types.json")
    kg_syms = KGSymbols(entity_dump, emb_dir, "kg_adj.txt")
    a2q= json.load(open(os.path.join(input_dir, "entity_db/entity_mappings/alias2qids.json")))
    with open(f"{emb_dir}/qid2desc.json") as f:
        qid2dec = json.load(f)
    return input_dir, emb_dir, entity_dump, types_wd, types_rel, kg_syms, a2q, qid2dec


def load_bootleg_relation_map():
    emb_dir = "" # fill in directory for Bootleg embeddding data (embs)
    with open (f'{emb_dir}/embs/pid_vocab.json") as f:
        pid_names = json.load(f)
    return pid_names


def save_datasets_stats(modified_datasets, dir_version):
    if not os.path.exists(dir_version):
        os.makedirs(dir_version)
    for split, md in modified_datasets.items():
        with open(f"{dir_version}/{split}_aug.json", 'w') as f:
            for i in range(len(md)):
                json.dump(md[i], f)
                if i < len(md) - 1:
                    f.write('\n')
            print(f"Saved {split}.")

    print(f"Saved all results to directory: {dir_version}")


def load_fewrel(save_path):
    def get_label_map(datasets):
        label_map = {}
        for name, dataset in datasets.items():
            for entry in dataset:
                label_no = entry['label_no']
                label = entry['label']
                label_map[label] = label_no
        return label_map

    if not os.file_path.exists(f"{save_path}/train_aug.json"):
        os.makedirs(save_path)
        datasets = {}
        dataset_names = ['train', 'dev', 'test']
        dataset_files = ['train_aug.json', 'dev_aug.json', 'test_aug.json']
        for name, file in zip(dataset_names, dataset_files):
            path = f"{data_dir}/{file}"
            dataset = load_jsonl(path)
            datasets[name] = dataset
    else:
        datasets = {}
        dataset_names = ['train', 'dev', 'test']
        dataset_files = ['train_aug.json', 'dev_aug.json', 'test_aug.json']
        unq_idx = 0
        for name, file in zip(dataset_names, dataset_files):
            path = f"{save_path}/{file}"
            dataset = load_jsonl(path)

            reformat_dataset = []
            for ex in dataset:
                new_ex = ex.copy()

                if 'label_no' not in ex.keys():
                    idx = label_map[new_ex['label']]
                    new_ex['label_no'] = idx
                if 'unq_idx' not in new_ex.keys():
                    new_ex['unq_idx'] = unq_idx
                    unq_idx += 1
                if 'label_name' not in new_ex.keys():
                    rel_name = new_ex['label']
                    new_ex['label_name'] = rel_name

                new_ex['ents'] = {}
                new_ex['ents']['subj'] = {
                    'qid': ex['ents'][0][0],
                    'start': ex['ents'][0][1],
                    'end': ex['ents'][0][2]
                }

                new_ex['ents']['obj'] = {
                    'qid': ex['ents'][1][0],
                    'start': ex['ents'][1][1],
                    'end': ex['ents'][1][2]
                }
                reformat_dataset.append(new_ex)

                # reset
                new_ex['pow2_entropy_ranked_obj'] = []
                new_ex['pop_ranked_obj'] = []
                new_ex['pow2_entropy_ranked_subj'] = []
                new_ex['pop_ranked_subj'] = []

            datasets[name] = reformat_dataset

    label_map = get_label_map(datasets)

    return datasets, label_map

