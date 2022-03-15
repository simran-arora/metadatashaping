import os
import torch
import transformers
from transformers import AutoModel, AutoModelForMaskedLM
from torch import nn
import torch.nn.functional as F

class BertModule(nn.Module):
    def __init__(self, args, tokenizer_len, cache_dir="./cache/"):
        super().__init__()
        self.max_seq_len = args.max_seq_length
        bert_model_name = args.bert_model

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.transformer = AutoModel.from_pretrained(
            bert_model_name,
            output_attentions=True,
            cache_dir=cache_dir
        )
        self.transformer.resize_token_embeddings(tokenizer_len)

    def forward(self, token_ids, token_segments=None, attention_mask=None, split="train"):
        enc_layer, pooled_output, attntuple = self.transformer(token_ids,
                                                               token_type_ids=token_segments,
                                                               attention_mask=attention_mask,
                                                               output_attentions=True,
                                                               return_dict=False)
        return enc_layer, pooled_output


class PredHead(nn.Module):
    """ Train a linear classifier from the "sentence representation emb." to classes """
    def __init__(self, bert_output_dim, nclasses, tokenizer_len):
        super().__init__()
        self.m = nn.Linear(bert_output_dim, nclasses)

    def forward(self, input):
        output = self.m(input)
        return output

