import json
import os
import jsonlines
from tqdm import tqdm
import numpy as np


def write_to_file(file_path, value):
    """
    Write value to file.
    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    if not isinstance(value, str):
        value = str(value)
    fout = open(file_path, "w")
    fout.write(value + "\n")
    fout.close()


def write_to_json_file(file_path, dict):
    """
    Write dict to json file.
    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    for k, v in dict.items():
        if type(v) == np.float:
            json_obj = json.dumps(eval(str(dict)))
        else:
            json_obj = json.dumps(dict)
        break
    fout = open(file_path, "w")
    fout.write(json_obj)
    fout.close()


def load_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def load_jsonl(file_path):
    data = []
    with open(file_path) as jsonl_file:
        for line in jsonl_file:
            val = json.loads(line)
            data.append(val)
    print(len(data))
    return data


def load_new_toks(file_path):
    new_toks = []
    with open(file_path) as f:
        input = f.read()
        input = input.split('\n')
        for tok in input:
            if tok:
                new_toks.append(tok)
    return new_toks


def save_args(args, logdir):
    with open(f'{logdir}/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f)
