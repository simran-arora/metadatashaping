import argparse
import logging
import sys
import os
import torch
import json
import random
import numpy
import pickle

import emmental
from data import create_dataloaders, load_data
from emmental import Meta
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.utils.parse_args import parse_args, parse_args_to_config
from emmental.utils.utils import nullable_string, str2bool, str2list, nullable_int
import task
from utils import write_to_file, write_to_json_file, load_new_toks
import transformers
from transformers import AutoTokenizer
from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import AutoModel, AutoModelForMaskedLM
from task_config import MARK_TOKS

logger = logging.getLogger(__name__)


def add_application_args(parser):

    # Application configuration
    application_config = parser.add_argument_group("Application configuration")

    application_config.add_argument(
        "--task", type=str, required=True,
        help="Text classification tasks"
    )
    application_config.add_argument(
        "--task_type", type=str, required="classification",
        choices=["classification"],
        help="Types of tasks"
    )
    application_config.add_argument(
        "--data_dir", type=str, default="data", help="The path to dataset"
    )
    application_config.add_argument(
        "--new_tok_file", type=str, default="",
        help="Synthetic dataset new tokens filename"
    )
    application_config.add_argument(
        "--train", type=int, default=1,
        help="just running eval"
    )
    application_config.add_argument(
        "--debugmode", type=int, default=0,
        help="whether to use debugging shortcuts"
    )
    application_config.add_argument(
        "--write_datashaped_data", type=bool, default=False,
        help="whether to save the shaped dataset"
    )

    application_config.add_argument(
        "--bert_model",
        type=str,
        default="bert-base-uncased",
        choices=["bert-base-cased", "bert-base-uncased"],
        help="Which bert pretrained model to use",
    )

    application_config.add_argument(
        "--max_seq_length", type=int, default=512, help="Maximum sentence length"
    )

    application_config.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["transformer"],
        help="Which model to use",
    )

    # HYPERPARAMETERS
    application_config.add_argument(
        "--batch_size", type=int, default=32, help="batch size"
    )
    application_config.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout"
    )
    application_config.add_argument(
        "--valid_batch_size",
        type=nullable_int,
        default=None,
        help="Validation batch size",
    )

    application_config.add_argument(
        "--fix_emb", type=str2bool, default=False, help="Fix word embedding or not"
    )

    # DATA SHAPING
    application_config.add_argument(
        "--use_mask", type=str2bool, default=False, help="Use masking or not"
    )
    application_config.add_argument(
        "--mlm_probability", type=float, default=0.15, help="percent of tokens to mask"
    )
    application_config.add_argument(
        "--using_metainfo", type=str, default=None, help="Use meta information"
    )
    application_config.add_argument(
        "--using_description", type=str, default=None, help="Use meta information"
    )
    application_config.add_argument(
        "--use_mark_tokens", type=str2bool, default=False, help="e.g. [SUBJ_START], [ENT]"
    )
    application_config.add_argument(
        "--use_bootleg", type=str2bool, default=False, help="use bootleg tagged meta info"
    )
    application_config.add_argument(
        "--use_entity_masking", type=str2bool, default=False, help="mask out the entity string"
    )
    application_config.add_argument(
        "--use_type_mask", type=str2bool, default=False, help="mask type words as regularization"
    )
    application_config.add_argument(
        "--subj_type_combo", type=str, default='prefer_relations', help="choice of types for subj"
    )
    application_config.add_argument(
        "--obj_type_combo", type=str, default='bootleg_only', help="choice of types for obj"
    )
    application_config.add_argument(
        "--subj_max_types", type=int, default=5, help="max number of types for subj"
    )
    application_config.add_argument(
        "--obj_max_types", type=int, default=5, help="max number of types for obj"
    )
    application_config.add_argument(
        "--description_max_len", type=int, default=5, help="max description words"
    )
    application_config.add_argument(
        "--use_aug_mark_tokens", type=str2bool, default=False, help="use marker to singal added information"
    )


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        "Text Classification Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser = parse_args(parser=parser)
    add_application_args(parser)

    args = parser.parse_args()
    config = parse_args_to_config(args)

    emmental.init(config["meta_config"]["log_path"], config=config)

    # Log configuration into filesup
    cmd_msg = " ".join(sys.argv)
    logger.info(f"COMMAND: {cmd_msg}")
    write_to_file(f"{Meta.log_path}/cmd.txt", cmd_msg)

    logger.info(f"Config: {Meta.config}")
    write_to_file(f"{Meta.log_path}/config.txt", Meta.config)

    datasets = {}
    data = []

    tokenizer = None
    if "transformer" in args.model or args.embedding == "bert":
        print("BERT TOKENIZER")

        # get special tokens
        mark_special_toks = []
        for word in MARK_TOKS:
            word = word.lower().strip(' ')
            mark_special_toks.append(word)

        if 1:
            tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
            print(f"SPECIAL TOKENS: {mark_special_toks}")
            special_tokens_dict = {'additional_special_tokens': mark_special_toks}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            logger.info(
                f"Added {num_added_toks} special tokens to the tokenizer vocabulary."
            )
        if not os.path.exists("./models/tokenizer/"):
            logger.info(f"Saving the tokenizer!")
            tokenizer.save_pretrained("./models/tokenizer/")

    # create datasets and data loaders
    task_name = args.task
    sortedKeys = sorted(tokenizer.get_vocab().keys())
    tokenizer_vocab = sortedKeys
    dataset = load_data(args, task_name, tokenizer_vocab=tokenizer_vocab, log_path=Meta.log_path)
    datasets[task_name] = dataset

    # create data loaders
    dataloaders = []
    dataloaders += create_dataloaders(
        task_name, datasets[task_name], args, tokenizer=tokenizer
    )

    if 'transformer' in args.model:
        # Specify parameter group for Adam BERT
        def grouped_parameters(model):
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            return [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": emmental.Meta.config["learner_config"][
                            "optimizer_config"
                    ]["l2"],
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        emmental.Meta.config["learner_config"]["optimizer_config"][
            "parameters"
        ] = grouped_parameters

    # create tasks
    tasks = {}
    if args.task_type == "classification":
        tasks[task_name] = task.create_task(
                task_name, args,
                datasets[task_name]["nclasses"],
                tokenizer=tokenizer
        )
    else:
        assert 0, print("Error with the task type.")

    # create tasks
    model = EmmentalModel(name="TC_task")
    for task_name, task in tasks.items():
        model.add_task(task)
    if Meta.config["model_config"]["model_path"]:
        model.load(Meta.config["model_config"]["model_path"])

    if args.train:
        emmental_learner = EmmentalLearner()
        emmental_learner.learn(model, dataloaders)

    # save metrics and models
    scores = model.score(dataloaders[1:])

    if args.checkpointing and args.train:
        logger.info(
            f"Best metrics: "
            f"{emmental_learner.logging_manager.checkpointer.best_metric_dict}"
        )
        write_to_file(
            f"{Meta.log_path}/best_metrics.txt",
            emmental_learner.logging_manager.checkpointer.best_metric_dict,
        )

    # Save metrics and models
    logger.info(f"Metrics: {scores}")
    scores["log_path"] = emmental.Meta.log_path
    write_to_json_file(f"{emmental.Meta.log_path}/metrics.txt", scores)

    if args.train:
        model.save(f"{emmental.Meta.log_path}/last_model.pth")

    if args.task_type == "classification":
        if task_name in ["fewrel", "tacred_revised"]:
            for dataloader in dataloaders:
                    if dataloader.split == "train":
                        continue
                    preds = model.predict(dataloader, return_preds=True)
                    res = ""
                    for uid, pred in zip(preds["uids"][task_name], preds["preds"][task_name]):
                        res += f"{uid}\t{pred}\n"
                    write_to_file(
                        f"{emmental.Meta.log_path}/{dataloader.split}_predictions.txt", res.strip()
                    )

                    res = ""
                    for uid, pred in zip(preds["uids"][task_name], preds["golds"][task_name]):
                        res += f"{uid}\t{pred}\n"
                    write_to_file(
                        f"{emmental.Meta.log_path}/{dataloader.split}_golds.txt", res.strip()
                    )

                    res = ""
                    for uid, pred in zip(preds["uids"][task_name], preds["probs"][task_name]):
                        pred = [str(round(pr, 3)) for pr in pred]
                        res += f"{uid}\t{pred}\n"
                    write_to_file(
                        f"{emmental.Meta.log_path}/{dataloader.split}_probs.txt", res.strip()
                    )
        else:
            for dataloader in dataloaders:
                if dataloader.split == "train":
                    continue

                preds = model.predict(dataloader, return_preds=True)
                all_logits = preds["probs"][task_name]
                all_labels = preds["golds"][task_name]

                res = ""
                for uid, pred in zip(preds["uids"][task_name], preds["golds"][task_name]):
                    label_indexes = [i for i, v in enumerate(pred) if v > 0]
                    res += f"{uid}\t{label_indexes}\n"
                write_to_file(
                    f"{emmental.Meta.log_path}/{dataloader.split}_golds.txt", res.strip()
                )

                res = ""
                for uid, pred in zip(preds["uids"][task_name], preds["probs"][task_name]):
                    predicted_indexes = pred
                    res += f"{uid}\t{predicted_indexes}\n"
                write_to_file(
                    f"{emmental.Meta.log_path}/{dataloader.split}_predictions.txt", res.strip()
                )

