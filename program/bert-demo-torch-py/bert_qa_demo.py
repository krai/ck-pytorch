#!/usr/bin/env python3

import numpy as np
import os
import torch
from transformers import BertConfig, BertForQuestionAnswering, BertTokenizer

## See under the hood
#
DEBUG_LEVEL                     = int( os.getenv('CK_DEBUG', 0) )

## Model config and weights, loaded from the transformers' hub by name:
#
BERT_MODEL_NAME                 = os.environ['CK_BERT_MODEL_NAME']

## Text inputs (not optional):
#
BERT_DATA_CONTEXT_PATH          = os.environ['CK_BERT_DATA_CONTEXT_PATH']
BERT_DATA_QUESTIONS_PATH        = os.environ['CK_BERT_DATA_QUESTIONS_PATH']

## Enabling GPU if available and not disabled:
#
USE_CUDA                = torch.cuda.is_available() and (os.getenv('CK_DISABLE_CUDA', '0') in ('NO', 'no', 'OFF', 'off', '0'))
TORCH_DEVICE            = 'cuda:0' if USE_CUDA else 'cpu'
print("Torch execution device: "+TORCH_DEVICE)


print(f"Loading BERT model '{BERT_MODEL_NAME}' from the HuggingFace transformers' hub ...")
model = BertForQuestionAnswering.from_pretrained(BERT_MODEL_NAME)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, return_token_type_ids=True, return_attention_mask=False)
bert_config_obj = model.config
model.eval()
model.to(TORCH_DEVICE)
print(f"Vocabulary size: {bert_config_obj.vocab_size}")

with open(BERT_DATA_CONTEXT_PATH, 'r') as context_file:
    context = ''.join( context_file.readlines() ).rstrip()
print(f"\nContext taken from '{BERT_DATA_CONTEXT_PATH}':")
print(context)
print('-'*64 )

with open(BERT_DATA_QUESTIONS_PATH, 'r') as questions_file:
    questions = questions_file.readlines()
print(f"Questions taken from '{BERT_DATA_QUESTIONS_PATH}':\n")

with torch.no_grad():
    for i, question in enumerate( questions ):
        question = question.rstrip()
        print(f"Question_{i+1}: {question}")

        sample_encoding         = tokenizer.encode_plus( question, context )
        sample_input_ids        = sample_encoding["input_ids"]
        if DEBUG_LEVEL>0:
            print( len(sample_input_ids) )
            print(sample_encoding)
        if DEBUG_LEVEL>1:
            print(tokenizer.convert_ids_to_tokens(sample_input_ids))

        start_scores, end_scores = model.forward(
            input_ids=torch.LongTensor(sample_encoding["input_ids"]).unsqueeze(0).to(TORCH_DEVICE),
            token_type_ids=torch.LongTensor(sample_encoding["token_type_ids"]).unsqueeze(0).to(TORCH_DEVICE),
#            attention_mask=torch.LongTensor(sample_encoding["attention_mask"]).unsqueeze(0).to(TORCH_DEVICE),
        )

        answer_start, answer_stop = start_scores.argmax(), end_scores.argmax()
        answer_ids = sample_input_ids[answer_start:answer_stop + 1]
        answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids, skip_special_tokens=True)
        answer = tokenizer.convert_tokens_to_string(answer_tokens)

        if DEBUG_LEVEL>1:
            print(f"Answer_{i+1}: [{answer_start}..{answer_stop}] -> {answer_tokens} -> '{answer}'\n")
        elif DEBUG_LEVEL>0:
            print(f"Answer_{i+1}: [{answer_start}..{answer_stop}] -> {answer}\n")
        else:
            print(f"Answer_{i+1}: {answer}\n")

