
import json
import os
import pickle
import subprocess
import sys

BERT_CODE_ROOT=os.environ['CK_ENV_MLPERF_INFERENCE']+'/language/bert'
BERT_DATA_ROOT=os.environ['CK_BERT_DATA_ROOT']

sys.path.insert(0, BERT_CODE_ROOT)
sys.path.insert(0, BERT_CODE_ROOT + '/DeepLearningExamples/TensorFlow/LanguageModeling/BERT')

import numpy as np
import torch
from transformers import BertConfig, BertForQuestionAnswering, BertTokenizer
from create_squad_data import read_squad_examples, convert_examples_to_features

## Processing by batches:
#
BATCH_COUNT             = int(os.getenv('CK_BATCH_COUNT', 1))

## Enabling GPU if available and not disabled:
#
USE_CUDA                = torch.cuda.is_available() and (os.getenv('CK_DISABLE_CUDA', '0') in ('NO', 'no', 'OFF', 'off', '0'))
TORCH_DEVICE            = 'cuda:0' if USE_CUDA else 'cpu'

print("Torch execution device: "+TORCH_DEVICE)


cache_path='eval_features.pickle'
max_seq_length = 384
max_query_length = 64
doc_stride = 128

print("Loading BERT configs...")
with open(BERT_CODE_ROOT + '/bert_config.json') as f:
    bert_config_dict = json.load(f)

bert_config_obj = BertConfig( **bert_config_dict )

print("Loading PyTorch model...")
model = BertForQuestionAnswering(bert_config_obj)
model.eval()
model.to(TORCH_DEVICE)
model.load_state_dict(torch.load(BERT_DATA_ROOT +"/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"))


eval_features = []
# Load features if cached, convert from examples otherwise.
if os.path.exists(cache_path):
    print("Loading cached features from '%s'..." % cache_path)
    with open(cache_path, 'rb') as cache_file:
        eval_features = pickle.load(cache_file)
else:
    print("No cached features at '%s'... converting from examples..." % cache_path)

    print("Creating tokenizer...")
    tokenizer = BertTokenizer(BERT_DATA_ROOT + "/bert_tf_v1_1_large_fp32_384_v2/vocab.txt")

    print("Reading examples...")
    eval_examples = read_squad_examples(input_file=BERT_DATA_ROOT + "/dev-v1.1.json",
        is_training=False, version_2_with_negative=False)

    print("Converting examples to features...")
    def append_feature(feature):
        eval_features.append(feature)

    convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
        output_fn=append_feature,
        verbose_logging=False)

    print("Caching features at '%s'..." % cache_path)
    with open(cache_path, 'wb') as cache_file:
        pickle.dump(eval_features, cache_file)

TOTAL_EXAMPLES=len(eval_features)
print("Total examples available: {}".format(TOTAL_EXAMPLES))

## Processing by batches:
#
BATCH_COUNT             = int(os.getenv('CK_BATCH_COUNT')) or TOTAL_EXAMPLES

encoded_accuracy_log = []
with torch.no_grad():
    for i in range(BATCH_COUNT):
        selected_features = eval_features[i]

        start_scores, end_scores = model.forward(input_ids=torch.LongTensor(selected_features.input_ids).unsqueeze(0).to(TORCH_DEVICE),
            attention_mask=torch.LongTensor(selected_features.input_mask).unsqueeze(0).to(TORCH_DEVICE),
            token_type_ids=torch.LongTensor(selected_features.segment_ids).unsqueeze(0).to(TORCH_DEVICE))
        output = torch.stack([start_scores, end_scores], axis=-1).squeeze(0).cpu().numpy()

        encoded_accuracy_log.append({'qsl_idx': i, 'data': output.tobytes().hex()})
        print("Batch #{}/{} done".format(i+1, BATCH_COUNT))


with open('accuracy_log.json', 'w') as f:
    json.dump(encoded_accuracy_log, f)

cmd = "python3 "+BERT_CODE_ROOT+"/accuracy-squad.py --val_data={} --log_file=accuracy_log.json --out_file=predictions.json --max_examples={}".format(BERT_DATA_ROOT + "/dev-v1.1.json", BATCH_COUNT)
subprocess.check_call(cmd, shell=True)
