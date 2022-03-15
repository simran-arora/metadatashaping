from functools import partial

import torch.nn.functional as F
from torch import nn
from numpy import ndarray
from collections import Counter
from typing import Dict, List, Optional
import numpy as np

from emmental.modules.identity_module import IdentityModule
from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from bert_modules import BertModule
from task_config import IDTOLABEL

key = 'fewrel'
ID_TO_LABEL = IDTOLABEL[key]
NO_RELATION = -1

def bce_loss(module_name, immediate_output_dict, Y, active):
    logits = immediate_output_dict[module_name][0][active]
    if immediate_output_dict['_input_']['split'][0] != "train":
        return logits
    labels = Y[active].float()
    loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits))
    return loss


def bce_output(module_name, immediate_output_dict):
    return immediate_output_dict[module_name][0]


def ce_loss(module_name, immediate_output_dict, Y, active):
    return F.cross_entropy(
        immediate_output_dict[module_name][0][active], Y.view(-1)[active]
    )


def output(module_name, immediate_output_dict):
    return F.softmax(immediate_output_dict[module_name][0])


def tacred_scorer(
    golds: ndarray,
    probs: ndarray,
    preds: Optional[ndarray],
    uids: Optional[List[str]] = None,
) -> Dict[str, float]:
    
    res = {}
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(golds)):
        gold = golds[row]
        guess = preds[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    relations = gold_by_relation.keys()
    for relation in sorted(relations):
        # (compute the score)
        correct = correct_by_relation[relation]
        guessed = guessed_by_relation[relation]
        gold = gold_by_relation[relation]
        prec = 1.0
        if guessed > 0:
            prec = float(correct) / float(guessed)
        recall = 0.0
        if gold > 0:
            recall = float(correct) / float(gold)
        f1 = 0.0
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)

        res[f"{ID_TO_LABEL[relation]}_prec"] = prec
        res[f"{ID_TO_LABEL[relation]}_rec"] = recall
        res[f"{ID_TO_LABEL[relation]}_f1"] = f1

    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(
            sum(guessed_by_relation.values())
        )
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(
            sum(gold_by_relation.values())
        )
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    res["Precision"] = prec_micro
    res["Recall"] = recall_micro
    res["F1"] = f1_micro

    n_matches = np.where(golds == preds)[0].shape[0]

    res["Accuracy"] = n_matches / golds.shape[0]

    return res


def create_task(task_name, args, nclasses, tokenizer=None):

    print(f"MODEL: {args.model}")
    input_module = IdentityModule()
    d_out = 768 if "base" in args.bert_model else 1024
    feature_extractor = BertModule(args, len(tokenizer), cache_dir="./cache/")

    task_name = task_name
    task_flow = []

    module_pool = nn.ModuleDict(
        {
            "input": input_module,
            "feature": feature_extractor,
            "dropout": nn.Dropout(args.dropout),
            f"{task_name}_sigmoid": nn.Sigmoid(),
            f"{task_name}_pred_head": nn.Linear(d_out, nclasses)
        }
    )
    print(f"nclasses: {nclasses}")

    if args.model in ["transformer", "transformer_short"]:
        task_flow.extend(
            [
                {
                    "name": "feature",
                    "module": "feature",
                    "inputs": [
                        ("_input_", "sent_token_ids"),
                        ("_input_", "sent_token_segments"),
                        ("_input_", "sent_token_masks"),
                        ("_input_", "split")
                    ],
                },
                {
                    "name": f"{task_name}_pred_head",
                    "module": f"{task_name}_pred_head",
                    "inputs": [
                        ("feature", 1),
                    ],
                },
            ],
        )

    # LOSS FUNCTIONS
    loss_func = partial(ce_loss, f"{task_name}_pred_head")
    output_func = partial(output, f"{task_name}_pred_head")

    # SCORERS (obtained from ERNIE and TACRED source code)
    customize_metric_funcs = {
        "re_scorer": tacred_scorer,
    }
    scorer = Scorer(customize_metric_funcs=customize_metric_funcs)

    return EmmentalTask(
        name=task_name,
        module_pool=module_pool,
        task_flow=task_flow,
        loss_func=loss_func,
        output_func=output_func,
        scorer=scorer,
    )
