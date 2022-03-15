import logging
import torch
from dataset import TokenizedDataset
import json
import numpy as np

from dataloader import (
    load_fewrel,
)
from emmental.data import EmmentalDataLoader, EmmentalDataset

logger = logging.getLogger(__name__)

load_dataset = {
    "fewrel": load_fewrel,
}

def load_data(args, task, tokenizer_vocab=None, training_data_size=None, log_path=''):
    # Load data from file
    data_dir = args.data_dir
    train_x, train_y, valid_x, valid_y, test_x, test_y, train_unq_idx, dev_unq_idx, test_unq_idx, train_shaped, valid_shaped, test_shaped, nclasses = load_dataset[task](data_dir, args, token_vocab=tokenizer_vocab, seed=args.seed)

    if type(train_y[0]) == int:
        train_y = torch.LongTensor(train_y)
        valid_y = torch.LongTensor(valid_y)
        test_y = torch.LongTensor(test_y)

    dataset = {
        "train": (train_x, train_y, train_unq_idx),
        "valid": (valid_x, valid_y, dev_unq_idx),
        "test": (test_x, test_y, test_unq_idx),
        "nclasses": nclasses,
    }

    shaped_info = {
        'train': (train_shaped),
        'valid': (valid_shaped),
        'test': (test_shaped)
    }
    
    if args.write_datashaped_data and log_path:
        for split in ['train', 'valid', 'test']:
            x_lst, y_lst, unq_idx_lst = dataset[split]
            shape_tuples = shaped_info[split]

            shaped_dataset = []
            for x, y, unq_idx, shape_tup in zip(x_lst, y_lst, unq_idx_lst, shape_tuples):
                entry = {}
                entry['shape_text'] = x
                entry['label'] = y
                entry['unq_idx'] = unq_idx
                entry['text'] = shape_tup[0]
                entry['shape_info'] = shape_tup[1]
                shaped_dataset.append(entry)

            with open(f"{log_path}/{split}_shaped.jsonl", "w") as f:
                for entry in shaped_dataset:
                    json.dump(entry, f)
                    f.write("/n")

    return dataset


def create_dataloaders(task_name, dataset, args, oov="~#OoV#~", tokenizer=None):
    # Create dataloaders
    dataloaders = []

    for split in ["train", "valid", "test"]:
        split_x, split_y, unq_idx = dataset[split]

        if args.task_type == "classification":
            emmental_data = TokenizedDataset(
                args,
                args.task,
                split_x, split_y, unq_idx,
                tokenizer=tokenizer,
                split=split,
                max_seq_length=args.max_seq_length,
            )
        else:
            assert 0, print("Error with the task type.")

        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict={task_name: "label"},
                dataset=emmental_data,
                split=split,
                shuffle=True if split == "train" else False,
                batch_size=args.batch_size
                if split in args.train_split or args.valid_batch_size is None
                else args.valid_batch_size,
                num_workers=4,
            )
        )

        logger.info(
            f"Built dataloader for {split} set with {len(emmental_data)} "
            f"samples (Shuffle={split in args.train_split}, "
            f"Batch size={dataloaders[-1].batch_size})."
        )

    return dataloaders

