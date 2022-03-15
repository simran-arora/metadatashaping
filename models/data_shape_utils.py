import os
import random
import re
import math
import numpy as np
from utils import load_jsonl, load_json
import json
from task_config import SUBJ_START, SUBJ_END, OBJ_START, OBJ_END, ENT, ENT_START, ENT_END, AUG, ENT_MASK, CLS, SEP


def convert_token(tokens):
    clean_tokens = []
    for token in tokens:
        clean_token = token
        """ Convert PTB tokens to normal tokens """
        if token.lower() == "-lrb-":
            clean_token = "("
        elif token.lower() == "-rrb-":
            clean_token = ")"
        elif token.lower() == "-lsb-":
            clean_token = "["
        elif token.lower() == "-rsb-":
            clean_token = "]"
        elif token.lower() == "-lcb-":
            clean_token = "{"
        elif token.lower() == "-rcb-":
            clean_token = "}"
        clean_tokens.append(clean_token)
    return clean_tokens


def apply_mask_scheme(orig_toks, path, args, token_vocab=None):
    mlm_probability = args.mlm_probability
    if 'train' not in path:
        return orig_toks
    THRESHOLD = 5
    mask_scheme = "base_scheme"
    if mask_scheme == "base_scheme":
        sent_length = len(orig_toks)
        mask_num = math.ceil(sent_length * mlm_probability)
        mask = np.random.choice(sent_length, mask_num, replace=False)
        mask = np.array([m for m in mask.tolist()])

        toks = np.copy(orig_toks).tolist()
        mask = set(mask)
        for i in range(sent_length):
            if i in mask:
                rand = np.random.random()
                if rand < 0.8:
                    toks[i] = '[MASK]'
                elif rand < 0.9 and len(orig_toks) > THRESHOLD:
                    random_swap = np.random.choice(token_vocab)
                    toks[i] = str(random_swap)
    return toks


def insert_re_description(args, line, aug_tok, text, subj_str, obj_str, token_vocab=None):
    # get descriptions
    subj_desc = line['subj_desc'].strip(' ')
    obj_desc = line['obj_desc'].strip(' ')

    MAX_DESC_LEN = args.description_max_len
    subj_desc_toks = subj_desc.split(" ")[:MAX_DESC_LEN]
    subj_desc = " ".join(subj_desc_toks)
    obj_desc_toks = obj_desc.split(" ")[:MAX_DESC_LEN]
    obj_desc = " ".join(obj_desc_toks)

    MAX_DESC_LEN = args.description_max_len
    SIMILARITY_FILTER = 0  # only include description if it's not already similar to entity title
    if 'posstart' in args.using_description:
        if "subj" in args.using_description:
            text = aug_tok + subj_desc + aug_tok + text
        if "obj" in args.using_description:
            text = aug_tok + obj_desc + aug_tok + text
    elif 'posend' in args.using_description:
        if "subj" in args.using_description:
            text += " " + aug_tok + subj_desc + aug_tok
        if "obj" in args.using_description:
            text += " " + aug_tok + obj_desc + aug_tok
    elif 'posinsent' in args.using_description:
        if 'subj' in args.using_description and subj_desc:
            assert len(subj_desc) != 0, print(subj_desc)
            if MAX_DESC_LEN:
                subj_desc_toks = subj_desc.split()
                subj_desc = " ".join(subj_desc_toks[0:min(len(subj_desc_toks), MAX_DESC_LEN)])
                if len(subj_desc) == 0:
                    import pdb;
                    pdb.set_trace()
            assert len(subj_desc) != 0, print(subj_desc)
            overlap = len(set(subj_str.split()).intersection(set(subj_desc.split()))) / len(subj_desc)
            if SIMILARITY_FILTER and overlap > 0.75:
                subj_desc = ''

        if 'obj' in args.using_description and obj_desc:
            if MAX_DESC_LEN:
                obj_desc_toks = obj_desc.split()
                obj_desc = " ".join(obj_desc_toks[0:min(len(obj_desc_toks),MAX_DESC_LEN)])
            overlap = len(set(obj_str.split()).intersection(set(obj_desc.split()))) / len(obj_desc)
            if SIMILARITY_FILTER and overlap > 0.75:
                obj_desc = ''
    else:
        assert 0, print("missing position for description")
    return text, subj_desc, obj_desc


def assemble_obj_types(args, ex, path, token_vocab=None):
    # OBJECT
    type_combo = args.obj_type_combo
    obj_max_toks = args.obj_max_types
    obj_entropy_ranked = ex['pow2_entropy_ranked_obj'].copy()
    pop_ranked_obj = ex['pop_ranked_obj'].copy()

    if type_combo == 'entropy_ranked' and obj_entropy_ranked is not None:
        clean_types = obj_entropy_ranked.copy()
    elif type_combo == 'entropy_ranked_WORST' and obj_entropy_ranked is not None:
        clean_types = obj_entropy_ranked.copy()
        clean_types.reverse()
    elif type_combo == 'popularity_ranked' and pop_ranked_obj is not None:
        clean_types = pop_ranked_obj.copy()
    elif type_combo == 'popularity_ranked_shuffle' and pop_ranked_obj is not None:
        clean_types = pop_ranked_obj.copy()
        random.shuffle(clean_types)
    else:
        assert 0, print("invalid choice of type combo!")

    obj_types_toks = []
    for ty in clean_types[0:obj_max_toks]:
        obj_types_toks.extend(ty.split())

    if obj_types_toks:
        if args.use_type_mask:
            obj_types_toks = apply_mask_scheme(obj_types_toks, path, args, token_vocab=token_vocab)
            if 'train' not in path:
                assert '[MASK]' not in obj_types_toks
    return obj_types_toks


def assemble_subj_types(args, ex, path, token_vocab=None):
    # SUBJECT
    type_combo = args.subj_type_combo
    subj_max_toks = args.subj_max_types
    subj_entropy_ranked = ex['pow2_entropy_ranked_subj'].copy()
    pop_ranked_subj = ex['pop_ranked_subj'].copy()

    if type_combo == 'entropy_ranked' and subj_entropy_ranked is not None:
        clean_types = subj_entropy_ranked.copy()
    elif type_combo == 'entropy_ranked_WORST' and subj_entropy_ranked is not None:
        clean_types = subj_entropy_ranked.copy()
        clean_types.reverse()
    elif type_combo == 'popularity_ranked' and pop_ranked_subj is not None:
        clean_types = pop_ranked_subj.copy()
    elif type_combo == 'popularity_ranked_shuffle' and pop_ranked_subj is not None:
        clean_types = pop_ranked_subj.copy()
        random.shuffle(clean_types)
    else:
        assert 0, print("invalid choice of type combo for subject!")

    subj_types_toks = []
    for ty in clean_types[0:subj_max_toks]:
        subj_types_toks.extend(ty.split())

    if subj_types_toks:
        if args.use_type_mask:
            subj_types_toks = apply_mask_scheme(subj_types_toks, path, args, token_vocab=token_vocab)
            if 'train' not in path:
                assert '[MASK]' not in subj_types_toks
    return subj_types_toks


def insert_re_metainfo(args, line, path, token_vocab=None):
    subj_types_str = ''
    obj_types_str = ''
    if 'subj' in args.using_metainfo:
        subj_types_toks = assemble_subj_types(args, line, path, token_vocab=token_vocab)
        subj_types_str = " ".join(subj_types_toks)
    if 'obj' in args.using_metainfo:
        obj_types_toks = assemble_obj_types(args, line, path, token_vocab=token_vocab)
        obj_types_str = " ".join(obj_types_toks)
    return subj_types_str, obj_types_str
