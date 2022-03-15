import os
import random
import re
import math
import ast
from tqdm import tqdm

import numpy as np
from utils import load_jsonl, load_json
from data_shape_utils import convert_token, apply_mask_scheme
from data_shape_utils import insert_re_description, insert_re_metainfo
import json

from task_config import SUBJ_START, SUBJ_END, OBJ_START, OBJ_END, ENT, ENT_START, ENT_END, AUG, ENT_MASK, CLS, SEP


def load_corpus_re(path, args, token_vocab=None, seed=1234):
    data = []
    labels = []
    shaped_spans_lst = []

    try:
        dataset = load_json(path)
    except:
        dataset = load_jsonl(path)
    unq_idx = []

    AUG_TY = " "
    if args.use_aug_mark_tokens:
        AUG_TY = AUG

    for i, line in enumerate(dataset):
        shaped_spans = []
        label = line['label_no']
        if "tacred" in args.task:
            label = line['revised_label_no']
        text = line['text']
        unq_id = line['unq_idx']
        used_other = 0

        # USING ENTITY TITLE
        subj_start = line['ents']['subj']['start']
        subj_end = line['ents']['subj']['end']
        obj_start = line['ents']['obj']['start']
        obj_end = line['ents']['obj']['end']
        if subj_start == 1:
            subj_start = 0
        if obj_start == 1:
            obj_start = 0

        orig_subj = line['text'][subj_start:subj_end]
        orig_obj = line['text'][obj_start:obj_end]
        subj_entity_title = orig_subj
        obj_entity_title = orig_obj

        # SUBJSTART, SUBJEND TOKENS
        if args.use_mark_tokens:
            subj = line['text'][subj_start:subj_end]
            obj = line['text'][obj_start:obj_end]

            # ADD METAINFO AND DESCRIPTION IN MIDDLE OF SENTENCE
            subj_desc_str, obj_desc_str = "", ""
            subj_tystr, obj_tystr = "", ""
            if 'posinsent' in args.using_metainfo:
                subj_tystr, obj_tystr = insert_re_metainfo(args, line, path, token_vocab=token_vocab)
            if 'posinsent' in args.using_description:
                _, subj_desc_str, obj_desc_str = insert_re_description(args, line, AUG_TY, text, subj, obj, token_vocab=token_vocab)
            SUBJ_END_TY = SUBJ_END
            OBJ_END_TY = OBJ_END

            subj_metainfo_str = ""
            if subj_tystr and subj_desc_str:
                subj_metainfo_str = subj_tystr + " " + subj_desc_str
            elif subj_tystr:
                subj_metainfo_str = subj_tystr
            elif subj_desc_str:
                subj_metainfo_str = subj_desc_str
            if subj_tystr or subj_desc_str:
                subj_metainfo_toks = subj_metainfo_str.split()
                sub_str = " ".join(subj_metainfo_toks)
                SUBJ_END_TY = AUG_TY + sub_str + AUG_TY + SUBJ_END.strip()

            obj_metainfo_str = ""
            if obj_tystr and obj_desc_str:
                obj_metainfo_str = obj_tystr + " " + obj_desc_str
            elif obj_tystr:
                obj_metainfo_str = obj_tystr
            elif obj_desc_str:
                obj_metainfo_str = obj_desc_str
            if obj_tystr or obj_desc_str:
                obj_metainfo_toks = obj_metainfo_str.split()
                sub_str = " ".join(obj_metainfo_toks)
                OBJ_END_TY = AUG_TY + sub_str + AUG_TY + OBJ_END.strip()

            # need to adjust to the subj/obj start/end positions as we insert toks
            if subj and obj and obj_start > subj_start:
                obj_start += len(SUBJ_START) + len(SUBJ_END_TY) + (len(subj)-len(orig_subj))
                obj_end += len(SUBJ_START) + len(SUBJ_END_TY) + (len(subj)-len(orig_subj))
            
            if subj:
                text = text[:subj_start] + SUBJ_START + subj + SUBJ_END_TY + text[subj_end:]
                subj_start_span = (subj_start, subj_start + len(SUBJ_START)) # START_SUBJ
                subj_end_span = (subj_start + len(SUBJ_START) + len(subj),
                                 subj_start + len(SUBJ_START) + len(subj) + len(SUBJ_END_TY))  # END_SUBJ
                subj_start = subj_start + len(SUBJ_START)
                subj_end = subj_start + len(subj)
                assert subj == text[subj_start:subj_end]
            if obj:
                text = text[:obj_start] + OBJ_START + obj + OBJ_END_TY + text[obj_end:]
                shaped_spans.append((obj_start, obj_start + len(OBJ_START))) # START_OBJ
                shaped_spans.append((obj_start + len(OBJ_START) + len(obj), obj_start + len(OBJ_START) + len(obj) + len(OBJ_END_TY))) # END_OBJ
                obj_start = obj_start + len(OBJ_START)
                obj_end = + obj_start + len(obj)
                assert obj == text[obj_start:obj_end]
            
            if obj_start < subj_start and subj:
                shift_start = len(OBJ_START) + len(OBJ_END_TY) + (len(obj)-len(orig_obj))
                subj_start += shift_start
                subj_end += shift_start
                subj_start_span = (subj_start_span[0] + shift_start, subj_start_span[1] + shift_start)  # START_SUBJ
                subj_end_span = (subj_end_span[0] + shift_start,   subj_end_span[1] + shift_start)  # END_SUBJ

            shaped_spans.append(subj_start_span)
            shaped_spans.append(subj_end_span)

        # ADD METAINFO AND DESCRIPTION AT START/END OF SENTENCE
        if args.using_description != None and args.using_description != "None" and 'posinsent' not in args.using_description:
            text, _, _ = insert_re_description(args, line, AUG_TY, text, subj, obj)

        # put meta info at the start or end of the sentence
        if args.using_metainfo != None and args.using_metainfo != "None" and "posinsent" not in args.using_metainfo:
            subj_types_str, obj_types_str = insert_re_metainfo(args, line, path, token_vocab=token_vocab)
            subj_metainfo_str = AUG_TY + subj_types_str + AUG_TY + AUG_TY + obj_types_str + AUG_TY
            text = text + subj_metainfo_str

        labels.append(label)
        clean_text = convert_token(text.split())
        shaped_spans_lst.append((line[f"text"], shaped_spans))

        if args.use_mark_tokens:
            assert len(shaped_spans) == 4
            assert SUBJ_START.strip() in clean_text, print(i, clean_text)
            assert SUBJ_END.strip() in clean_text, print(i, clean_text)
            assert OBJ_START.strip() in clean_text, print(i, clean_text)
            assert OBJ_END.strip() in clean_text, print(i, clean_text)
        data.append(clean_text)
        unq_idx.append(unq_id)
        if i % 20000 == 0:
            print(clean_text)
            print(label)
            print(used_other)
            print()

    # shuffle the training data
    if 'train' in path and seed > 0:
        random.seed(seed)
        perm = list(range(len(data)))
        random.shuffle(perm)
        data = [data[i] for i in perm]
        labels = [labels[i] for i in perm]
        shaped_spans_lst = [shaped_spans_lst[i] for i in perm]
        unq_idx = [unq_idx[i] for i in perm]
    return data, labels, unq_idx, shaped_spans_lst


def load_fewrel(path, args, token_vocab=None, seed=1234):
    train = os.path.join(path, "train.json")
    test = os.path.join(path, "test.json")
    dev = os.path.join(path, "dev.json")

    train_data, train_labels, train_unq_idx, train_shaped = load_corpus_re(train, args, token_vocab=token_vocab, seed=seed)
    test_data, test_labels, test_unq_idx, test_shaped = load_corpus_re(test, args, token_vocab=token_vocab, seed=seed)
    dev_data, dev_labels, dev_unq_idx, dev_shaped = load_corpus_re(dev, args, token_vocab=token_vocab, seed=seed)
    nclasses = max(train_labels) + 1

    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels, train_unq_idx, dev_unq_idx, test_unq_idx, train_shaped, dev_shaped, test_shaped, nclasses


