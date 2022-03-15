import logging
from collections import Counter
import math
import numpy as np
import torch
import math
from emmental.data import EmmentalDataset
from tqdm import tqdm
import random
from random import randrange
from task_config import MARK_TOKS

logger = logging.getLogger(__name__)

CLS = "[CLS]"
SEP = "[SEP]"
MASK = '[MASK]'
PAD = '[PAD]'
pad_id = 0
mask_id = 1


def apply_mask_scheme(mlm_probability, orig_toks, label, task, added_tokens, vocab_tokens):
    mask_scheme = "base_scheme"
    if mask_scheme == "base_scheme":
        sent_length = len(orig_toks)
        mask_num = math.ceil(sent_length * mlm_probability)
        mask = np.random.choice(sent_length, mask_num, replace=False)
        mask = np.array(
            [m for m in mask.tolist() if orig_toks[m] not in added_tokens and orig_toks[m] != CLS and orig_toks[m] != SEP]
        )
        toks = np.copy(orig_toks)
        mask = set(mask)
        for i in range(sent_length):
            if i in mask:
                rand = np.random.random()
                if rand < 0.8:
                    toks[i] = MASK
                elif rand < 0.9:
                    random_swap = np.random.choice(vocab_tokens)
                    if random_swap not in added_tokens:
                        toks[i] = random_swap
        toks = toks.tolist()
    return orig_toks, toks


def tokenize_on_the_fly(toks, label, max_seq_length, mlm_probability,
                        added_tokens, vocab_tokens, mark_tokens, tokenizer, tokenizer_cache,
                        use_mask, split, task):
    tokens = [CLS]
    masked_tokens = [CLS]
    num_tokens = 0
    num_fit_examples = 0

    # masked toks is the input, toks is labels of the masked words
    masked_toks = toks
    if use_mask and split == 'train':
        toks, masked_toks = apply_mask_scheme(mlm_probability, toks, label, task, added_tokens, vocab_tokens)

    # Tokenize
    for token, masked_token in zip(toks, masked_toks):
        if token == "[sep]":
            token = SEP
            masked_token = SEP
        if token == "[mask]":
            token = MASK
            masked_token = MASK
        if token in tokenizer_cache:
            sub_tokens = tokenizer_cache[token]
        elif token in mark_tokens:
            sub_tokens = [mark_tokens[token]]
            tokenizer_cache[token] = sub_tokens
        else:
            sub_tokens = tokenizer.tokenize(token)
            tokenizer_cache[token] = sub_tokens
        tokens.extend(sub_tokens)

        if token == masked_token:
            masked_tokens.extend(sub_tokens)
        else:
            masked_tokens.extend(tokenizer.tokenize(masked_token))

    tokens.append(SEP)
    masked_tokens.append(SEP)

    num_tokens += len(masked_tokens)
    truncated = 0
    if len(masked_tokens) > max_seq_length or len(tokens) > max_seq_length:
        # assert 0, print("exceeding max sequence length")
        masked_tokens = masked_tokens[:max_seq_length]
        tokens = tokens[:max_seq_length]
        truncated = 1
    else:
        num_fit_examples += 1

    segment_ids = [0] * len(masked_tokens)
    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    input_mask = [1] * len(input_ids)
    padding = [pad_id] * (max_seq_length - len(input_ids))

    input_mask += [0] * (max_seq_length - len(input_ids))  #padding
    segment_ids += [0] * (max_seq_length - len(input_ids))  #padding
    input_ids += padding
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    input_ids = torch.LongTensor(input_ids)
    input_mask = torch.LongTensor(input_mask)
    segment_ids = torch.LongTensor(segment_ids)

    mlm_label_ids = tokenizer.convert_tokens_to_ids(tokens)
    padding_mask_labels = [pad_id] * (max_seq_length - len(mlm_label_ids))
    mlm_label_ids += padding_mask_labels
    assert len(mlm_label_ids) == max_seq_length
    mlm_label_ids = torch.LongTensor(mlm_label_ids)

    return input_ids, mlm_label_ids, input_mask, segment_ids, tokenizer_cache, truncated


class InputExample(object):
    """A single training/test example."""

    def __init__(
            self,
            guid,
            sentence,
            label,
    ):
        self.guid = guid
        self.sentence = sentence
        self.label = label


def create_examples(args, split_x, split_y, unq_idx):
    """Creates examples."""
    examples = []

    for example, y, unq_id in zip(split_x, split_y, unq_idx):
        if 'uncased' in args.bert_model:
            tokens = [token.lower() for token in example if token not in [CLS, SEP]]
        else:
            tokens = [token for token in example if token not in [CLS, SEP]]
        examples.append(
            InputExample(
                guid=unq_id,
                sentence=tokens,
                label=y
            )
        )
    return examples


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            guid,
            sent_ids,
            sent_mask,
            sent_segment_ids,
            label,
            MLM_label
    ):
        self.guid = guid
        self.sent_ids = sent_ids
        self.sent_mask = sent_mask
        self.sent_segment_ids = sent_segment_ids
        self.label = label
        self.MLM_label = MLM_label


def convert_examples_to_features(examples):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        features.append(
            InputFeatures(
                guid=example.guid,
                sent_ids=example.sentence,
                sent_mask=[],
                sent_segment_ids=[],
                label=example.label,
                MLM_label=[]
            )
        )

    return features


def get_labels(examples):
    count = Counter()
    if type(examples[0].label) != list:
        for example in examples:
            count[example.label] += 1
        logger.info("%d labels" % len(count))
    else:
        for example in examples:
            label_lst = example.label
            for i, lab in enumerate(label_lst):
                if lab == 1:
                    count[i] += 1


class TokenizedDataset(EmmentalDataset):
    """Dataset to load Type dataset."""

    def __init__(
            self,
            args,
            name,
            split_x,
            split_y,
            unq_idx,
            tokenizer,
            split="train",
            mode="text",
            max_seq_length=128,
            special_tokens={},
            bert_mode="base",
    ):

        X_dict, Y_dict = (
            {
                "guids": [],
                "sent_token_ids": [],
                "sent_token_masks": [],
                "sent_token_segments": [],
                "split": [],
                "mlm_label_ids": []
            },
            {"label": []},
        )

        examples = create_examples(args, split_x, split_y, unq_idx)
        logger.info(f"{split} set stats:")
        get_labels(examples)

        features = convert_examples_to_features(examples)

        # task
        self.task = args.task

        # for shuffling the choice of cands during __get_item__
        self.cls_id = tokenizer.convert_tokens_to_ids(CLS)
        self.max_seq_length = max_seq_length

        # for masking
        self.pad_id = tokenizer.convert_tokens_to_ids(PAD)
        self.sep_id = tokenizer.convert_tokens_to_ids(SEP)
        self.mask_id = tokenizer.convert_tokens_to_ids(MASK)
        try:
            sortedKeys = tokenizer.get_vocab().keys()
            added_sortedKeys = tokenizer.get_added_vocab().keys()
            self.vocab_tokens_words = sortedKeys
            self.added_tokens_words = added_sortedKeys
        except:
            self.vocab_tokens_words = []  # list(tokenizer.load_vocab().keys())
            self.added_tokens_words = []  #list(tokenizer.special_tokens)
        self.mlm_probability = args.mlm_probability
        self.use_mask = args.use_mask
        self.split = split
        self.tokenizer_cache = {}
        self.tokenizer = tokenizer
        self.truncated = 0
        self.mark_tokens = {}

        for i, feature in enumerate(features):
                X_dict["guids"].append(feature.guid)
                X_dict["sent_token_ids"].append(feature.sent_ids)
                X_dict["sent_token_masks"].append(feature.sent_mask)
                X_dict["sent_token_segments"].append(feature.sent_segment_ids)
                Y_dict["label"].append(feature.label)
                X_dict["split"].append(split)
                X_dict["mlm_label_ids"].append(feature.MLM_label)

        arr = np.array(Y_dict["label"])
        Y_dict["label"] = torch.from_numpy(arr)

        logger.info(
            "%d (%.2f %%) examples cannot fit max_seq_length = %d"
            % (self.truncated, self.truncated * 100.0 / len(examples), max_seq_length)
        )

        super().__init__(name, X_dict=X_dict, Y_dict=Y_dict, uid="guids")

    def __getitem__(self, index):
        r"""Get item by index.

        Args:
          index(index): The index of the item.

        Returns:
          Tuple[Dict[str, Any], Dict[str, Tensor]]: Tuple of x_dict and y_dict

        """
        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}
        y_dict = {name: label[index] for name, label in self.Y_dict.items()}

        sent_ids, masked_label_ids, sent_masks, sent_segments, tokenizer_cache, truncated = tokenize_on_the_fly(
                x_dict['sent_token_ids'],
                y_dict['label'],
                self.max_seq_length,
                self.mlm_probability,
                self.added_tokens_words,  self.vocab_tokens_words,
                self.mark_tokens,
                self.tokenizer, self.tokenizer_cache, self.use_mask,
                self.split, self.task
        )

        x_dict['sent_token_ids'] = sent_ids
        x_dict['sent_token_masks'] = sent_masks
        x_dict['sent_token_segments'] = sent_segments
        x_dict['mlm_label_ids'] = masked_label_ids
        self.tokenizer_cache = tokenizer_cache
        self.truncated += truncated

        if index == 100:
            print(self.tokenizer.convert_ids_to_tokens(np.array(x_dict['sent_token_ids'])))
            print(y_dict['label'])

        return x_dict, y_dict


